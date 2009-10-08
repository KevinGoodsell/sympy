""" Caching facility for SymPy """

class Cache(dict):
    '''Simple dict-based cache.'''

    def fetch_args(self, *args, **kwargs):
        key = self._convert_args(args, kwargs)
        return self[key]

    def add_args(self, value, *args, **kwargs):
        key = self._convert_args(args, kwargs)
        self[key] = value

    def _convert_args(self, args, kwargs):
        '''Convert args and kwargs into a single hashable key.'''
        kwargs = kwargs.items()
        kwargs.sort()
        return (args, tuple(kwargs))

class NullCache(Cache):
    '''Cache that never stores anything.'''

    def __setitem__(self, key, value):
        pass

    def setdefault(self, key, default=None):
        return default

    def update(self, *args, **kwargs):
        pass

    def fetch_args(self, *args, **kwargs):
        raise KeyError()

    def add_args(self, value, *args, **kwargs):
        pass

class DebugCache(Cache):
    '''Cache that never produces anything, but verifies all added items
    to ensure they are immutable and match previously added items with
    the same key.'''

    def __init__(self):
        Cache.__init__(self)
        # Items are stored in a separate dict so that nothing ever comes
        # out of the cache.
        self._cache = {}

    def __setitem__(self, key, value):
        self._checked_add(key, value)

    def setdefault(self, key, default=None):
        self._checked_add(key, default)
        return default

    def update(self, *args, **kwargs):
        if not args:
            it = kwargs.iteritems()
        elif len(args) == 1 and not kwargs:
            try:
                it = args[0].iteritems()
            except AttributeError:
                it = iter(args[0])
        else:
            raise TypeError('bad args to update')

        for (key, value) in it:
            self._checked_add(key, value)

    def fetch_args(self, *args, **kwargs):
        raise KeyError()

    def add_args(self, value, *args, **kwargs):
        key = self._convert_args(args, kwargs)
        self._checked_add(key, value)

    def _checked_add(self, key, value):
        cached = self._cache.setdefault(key, value)
        assert cached == value, 'cached value != new value'

        # Verify the immutability of value. Hashability roughly
        # corresponds to immutability.
        hash(value)

def _cache_factory():
    if _usecache == 'no':
        return NullCache()
    elif _usecache == 'debug':
        return DebugCache()
    else:
        return Cache()

class CacheRegistry(object):
    '''Maintains a registry of all caches that have been created.'''

    def __init__(self):
        self._registry = {} # { <key object>: (cache_obj, 'comment') }

    def new_cache(self, key, comment=None, factory=_cache_factory):
        '''Create and return a new cache object for the given key (which
        uniquely identifies a cache) and optional comment.'''
        cache = factory()
        self._registry[key] = (cache, comment)
        return cache

    def get_cache(self, key):
        '''Return an existing cache for the given key. Raises KeyError if
        the cache doesn't already exist. In general you should keep a
        reference to the cache rather than calling this frequently.'''
        (cache, comment) = self._registry[key]
        return cache

    def get_comment(self, key):
        '''Return the comment for the cache identified by key, or None
        if the cache has no comment. Raises KeyError if the cache doesn't
        exist.'''
        (cache, comment) = self._registry[key]
        return comment

    def clear(self):
        '''Clear all items from all caches.'''
        for (cache, comment) in self._registry.values():
            cache.clear()

    def print_cache(self, file=None):
        '''Print all caches to file, which defaults to sys.stdout. Intended
        for debugging only.'''
        if file is None:
            import sys
            file = sys.stdout

        for (ckey, (cache, comment)) in self._registry.items():
            if comment:
                header = '%s (%r)' % (comment, ckey)
            else:
                header = repr(ckey)

            print >> file, '=' * len(header)
            print >> file, header
            print >> file, '=' * len(header)

            width = 0
            text = []
            for (k, v) in cache.items():
                kstr = repr(k)
                vstr = repr(v)
                width = max(width, len(kstr))
                text.append((kstr, vstr))

            lines = ['%-*s : %s' % (width, k, v) for (k, v) in text]
            print >> file, '\n'.join(lines)

registry = CacheRegistry()

class CacheDecorator(object):
    def cache_factory(self):
        '''Return a new object derived from Cache. This exists to be
        overridden in decorators that use a non-default cache type.'''
        return _cache_factory()

    def make_cache(self, func):
        import inspect

        if inspect.ismethod(func):
            code = func.im_func.func_code
        else:
            code = func.func_code

        comment = 'cache for function %s at %s:%d' % (func.__name__,
            code.co_filename, code.co_firstlineno)

        cache = registry.new_cache(func, comment, self.cache_factory)
        func._cache_it_cache = cache
        return cache

    def annotate_wrapper(self, wrapper, func):
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__

    def __call__(self, func):
        cache = self.make_cache(func)

        # Two possible wrappers: one that takes any (hashable) args and one
        # that takes a single arg and can therefore skip the steps required
        # for the multi-arg case.
        def wrapper_args(*args, **kwargs):
            try:
                return cache.fetch_args(*args, **kwargs)
            except KeyError:
                pass

            val = func(*args, **kwargs)
            cache.add_args(val, *args, **kwargs)
            return val

        def wrapper_fast(arg):
            try:
                return cache[arg]
            except KeyError:
                pass

            val = func(arg)
            cache[arg] = val
            return val

        import inspect

        (args, varargs, keywords, defaults) = inspect.getargspec(func)
        if len(args) == 1 and varargs is None and keywords is None:
            # Simple single-argument function
            wrapper = wrapper_fast
        else:
            wrapper = wrapper_args

        self.annotate_wrapper(wrapper, func)

        return wrapper

class CacheDebugDecorator(CacheDecorator):
    def cache_factory(self):
        return DebugCache()

class CacheArgDecorator(CacheDecorator):
    def __init__(self, argnum):
        self.argnum = argnum

    def __call__(self, func):
        cache = self.make_cache(func)
        argnum = self.argnum

        def wrapper(*args, **kwargs):
            arg = args[argnum]
            try:
                return cache[arg]
            except KeyError:
                pass

            val = func(*args, **kwargs)
            cache[arg] = val
            return val

        self.annotate_wrapper(wrapper, func)

        return wrapper

class CacheNullDecorator(CacheDecorator):
    def __call__(self, func):
        return func

# These are specific decorators. Typically they should not be used directly,
# but in a pinch they can be. Normally the 'cacheit*' decorators should be used,
# which are set based on the cache options.
#
# Decorators used without args:
cache_decorator = CacheDecorator()
cache_null_decorator = CacheNullDecorator()
cache_debug_decorator = CacheDebugDecorator()
# Decorators used with args:
cache_arg_decorator = CacheArgDecorator

# TODO: refactor CACHE & friends into class?

# global cache registry:
CACHE = []  # [] of
            #    (item, {} or tuple of {})

def print_cache():
    """print cache content"""

    for item, cache in CACHE:
        item = str(item)
        head = '='*len(item)

        print head
        print item
        print head

        if not isinstance(cache, tuple):
            cache = (cache,)
            shown = False
        else:
            shown = True

        for i, kv in enumerate(cache):
            if shown:
                print '\n*** %i ***\n' % i

            for k, v in kv.iteritems():
                print '  %s :\t%s' % (k, v)

def clear_cache():
    """clear cache content"""
    for item, cache in CACHE:
        if not isinstance(cache, tuple):
            cache = (cache,)

        for kv in cache:
            kv.clear()

########################################

def __cacheit_nocache(func):
    return func

def __cacheit(func):
    """caching decorator.

       important: the result of cached function must be *immutable*


       Example
       -------

       @cacheit
       def f(a,b):
           return a+b


       @cacheit
       def f(a,b):
           return [a,b] # <-- WRONG, returns mutable object


       to force cacheit to check returned results mutability and consistency,
       set environment variable SYMPY_USE_CACHE to 'debug'
    """

    func._cache_it_cache = func_cache_it_cache = {}
    CACHE.append((func, func_cache_it_cache))

    def wrapper(*args, **kw_args):
        if kw_args:
            keys = kw_args.keys()
            keys.sort()
            items = [(k+'=',kw_args[k]) for k in keys]
            k = args + tuple(items)
        else:
            k = args
        try:
            return func_cache_it_cache[k]
        except KeyError:
            pass
        func_cache_it_cache[k] = r = func(*args, **kw_args)
        return r

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper

def __cacheit_debug(func):
    """cacheit + code to check cache consitency"""
    cfunc = __cacheit(func)

    def wrapper(*args, **kw_args):
        # always call function itself and compare it with cached version
        r1 = func (*args, **kw_args)
        r2 = cfunc(*args, **kw_args)

        # try to see if the result is immutable
        #
        # this works because:
        #
        # hash([1,2,3])         -> raise TypeError
        # hash({'a':1, 'b':2})  -> raise TypeError
        # hash((1,[2,3]))       -> raise TypeError
        #
        # hash((1,2,3))         -> just computes the hash
        hash(r1), hash(r2)

        # also see if returned values are the same
        assert r1 == r2

        return r1

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper

def __cacheit_nondummy(func):
    func._cache_it_cache = func_cache_it_cache = {}
    CACHE.append((func, func_cache_it_cache))

    def wrapper(*args, **kw_args):
        if kw_args:
            try:
                dummy = kw_args['dummy']
            except KeyError:
                dummy = None
            if dummy:
                return func(*args, **kw_args)
            keys = kw_args.keys()
            keys.sort()
            items = [(k+'=',kw_args[k]) for k in keys]
            k = args + tuple(items)
        else:
            k = args
        try:
            return func_cache_it_cache[k]
        except KeyError:
            pass
        func_cache_it_cache[k] = r = func(*args, **kw_args)
        return r

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper

class MemoizerArg:
    """ See Memoizer.
    """

    def __init__(self, allowed_types, converter = None, name = None):
        self._allowed_types = allowed_types
        self.converter = converter
        self.name = name

    def fix_allowed_types(self, have_been_here={}):
        from basic import C
        i = id(self)
        if have_been_here.get(i): return
        allowed_types = self._allowed_types
        if isinstance(allowed_types, str):
            self.allowed_types = getattr(C, allowed_types)
        elif isinstance(allowed_types, (tuple, list)):
            new_allowed_types = []
            for t in allowed_types:
                if isinstance(t, str):
                    t = getattr(C, t)
                new_allowed_types.append(t)
            self.allowed_types = tuple(new_allowed_types)
        else:
            self.allowed_types = allowed_types
        have_been_here[i] = True
        return

    def process(self, obj, func, index = None):
        if isinstance(obj, self.allowed_types):
            if self.converter is not None:
                obj = self.converter(obj)
            return obj
        func_src = '%s:%s:function %s' % (func.func_code.co_filename, func.func_code.co_firstlineno, func.func_name)
        if index is None:
            raise ValueError('%s return value must be of type %r but got %r' % (func_src, self.allowed_types, obj))
        if isinstance(index, (int,long)):
            raise ValueError('%s %s-th argument must be of type %r but got %r' % (func_src, index, self.allowed_types, obj))
        if isinstance(index, str):
            raise ValueError('%s %r keyword argument must be of type %r but got %r' % (func_src, index, self.allowed_types, obj))
        raise NotImplementedError(`index,type(index)`)

class Memoizer:
    """ Memoizer function decorator generator.

    Features:
      - checks that function arguments have allowed types
      - optionally apply converters to arguments
      - cache the results of function calls
      - optionally apply converter to function values

    Usage:

      @Memoizer(<allowed types for argument 0>,
                MemoizerArg(<allowed types for argument 1>),
                MemoizerArg(<allowed types for argument 2>, <convert argument before function call>),
                MemoizerArg(<allowed types for argument 3>, <convert argument before function call>, name=<kw argument name>),
                ...
                return_value_converter = <None or converter function, usually makes a copy>
                )
      def function(<arguments>, <kw_argumnets>):
          ...

    Details:
      - if allowed type is string object then there `C` must have attribute
        with the string name that is used as the allowed type --- this is needed
        for applying Memoizer decorator to Basic methods when Basic definition
        is not defined.

    Restrictions:
      - arguments must be immutable
      - when function values are mutable then one must use return_value_converter to
        deep copy the returned values

    Ref: http://en.wikipedia.org/wiki/Memoization
    """

    def __init__(self, *arg_templates, **kw_arg_templates):
        new_arg_templates = []
        for t in arg_templates:
            if not isinstance(t, MemoizerArg):
                t = MemoizerArg(t)
            new_arg_templates.append(t)
        self.arg_templates = tuple(new_arg_templates)
        return_value_converter = kw_arg_templates.pop('return_value_converter', None)
        self.kw_arg_templates = kw_arg_templates.copy()
        for template in self.arg_templates:
            if template.name is not None:
                self.kw_arg_templates[template.name] = template
        if return_value_converter is None:
            self.return_value_converter = lambda obj: obj
        else:
            self.return_value_converter = return_value_converter

    def fix_allowed_types(self, have_been_here={}):
        i = id(self)
        if have_been_here.get(i): return
        for t in self.arg_templates:
            t.fix_allowed_types()
        for k,t in self.kw_arg_templates.items():
            t.fix_allowed_types()
        have_been_here[i] = True

    def __call__(self, func):
        cache = {}
        value_cache = {}
        CACHE.append((func, (cache, value_cache)))

        def wrapper(*args, **kw_args):
            kw_items = tuple(kw_args.items())
            try:
                return self.return_value_converter(cache[args,kw_items])
            except KeyError:
                pass
            self.fix_allowed_types()
            new_args = tuple([template.process(a,func,i) for (a, template, i) in zip(args, self.arg_templates, range(len(args)))])
            assert len(args)==len(new_args)
            new_kw_args = {}
            for k, v in kw_items:
                template = self.kw_arg_templates[k]
                v = template.process(v, func, k)
                new_kw_args[k] = v
            new_kw_items = tuple(new_kw_args.items())
            try:
                return self.return_value_converter(cache[new_args, new_kw_items])
            except KeyError:
                r = func(*new_args, **new_kw_args)
                try:
                    try:
                        r = value_cache[r]
                    except KeyError:
                        value_cache[r] = r
                except TypeError:
                    pass
                cache[new_args, new_kw_items] = cache[args, kw_items] = r
                return self.return_value_converter(r)
        return wrapper


class Memoizer_nocache(Memoizer):

    def __call__(self, func):
        # XXX I would be happy just to return func, but we need to provide
        # argument convertion, and it is really needed for e.g. Real("0.5")
        def wrapper(*args, **kw_args):
            kw_items = tuple(kw_args.items())
            self.fix_allowed_types()
            new_args = tuple([template.process(a,func,i) for (a, template, i) in zip(args, self.arg_templates, range(len(args)))])
            assert len(args)==len(new_args)
            new_kw_args = {}
            for k, v in kw_items:
                template = self.kw_arg_templates[k]
                v = template.process(v, func, k)
                new_kw_args[k] = v

            r = func(*new_args, **new_kw_args)
            return self.return_value_converter(r)

        return wrapper



# SYMPY_USE_CACHE=yes/no/debug
import os
_usecache = os.getenv('SYMPY_USE_CACHE', 'yes').lower()

if _usecache == 'no':
    Memoizer            = Memoizer_nocache
    cacheit             = cache_null_decorator
    cacheit_arg         = cache_null_decorator
elif _usecache in ('yes', 'debug'):
    cacheit     = cache_decorator
    cacheit_arg = cache_arg_decorator
else:
    raise RuntimeError('unknown argument in SYMPY_USE_CACHE: %s' % _usecache)
