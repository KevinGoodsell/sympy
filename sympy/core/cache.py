""" Caching facility for SymPy """

# TODO:
# * Add full documentation and tests
# * Remove old caching code
# * Add cache statistics

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
        if kwargs:
            kwargs = kwargs.items()
            kwargs.sort()
            kwtuple = tuple(kwargs)
        else:
            kwtuple = ()
        return (args, kwtuple)

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

    def update(self, seq=(), **kwargs):
        if hasattr(seq, 'iteritems'):
            it = seq.iteritems()
        else:
            it = iter(seq)

        for (key, value) in it:
            self._checked_add(key, value)

        for (key, value) in kwargs:
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
        return cache

    def annotate_wrapper(self, wrapper, func):
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__

    def __call__(self, func):
        cache = self.make_cache(func)

        def wrapper(*args, **kwargs):
            try:
                return cache.fetch_args(*args, **kwargs)
            except KeyError:
                pass

            val = func(*args, **kwargs)
            cache.add_args(val, *args, **kwargs)
            return val

        self.annotate_wrapper(wrapper, func)
        wrapper._cacheit_cache = cache

        return wrapper

class CacheIntroDecorator(CacheDecorator):
    def __init__(self, *args):
        arg_names = []
        for arg in args:
            arg_names.extend(arg.split())

        self.arg_names = frozenset(arg_names)

    def __call__(self, func):
        cache = self.make_cache(func)

        (wrapper_name, wrapper_text) = self.build_wrapper(func)
        exec wrapper_text in locals()
        wrapper = eval(wrapper_name)

        self.annotate_wrapper(wrapper, func)
        wrapper._cacheit_cache = cache
        wrapper._cacheit_wrapper_src = wrapper_text

        return wrapper

    def build_wrapper(self, func):
        # Check the function arguments against the arguments given in the
        # constructor.
        import inspect
        (args, varargs, varkw, defaults) = inspect.getargspec(func)
        all_args = set(args)
        if varargs: all_args.add(varargs)
        if varkw: all_args.add(varkw)

        bad_args = self.arg_names - all_args
        if bad_args:
            raise ValueError('Unknown argument(s): %s' % ', '.join(bad_args))

        # This is the template for the wrapper function.
        func_template = (
            'def %(name)s%(argspec)s:\n'
            '%(kwargs_lines)s'
            '    try:\n'
            '        return cache[%(key)s]\n'
            '    except KeyError:\n'
            '        pass\n'
            '\n'
            '    val = func%(call)s\n'
            '    cache[%(key)s] = val\n'
            '    return val\n'
        )

        # This is the dict that will contain all the template keys.
        # The rest of this function is almost entirely filling this in.
        filler = {}

        filler['name'] = func.__name__ + '_wrapper'
        filler['argspec'] = inspect.formatargspec(args, varargs, varkw, defaults)
        filler['call'] = inspect.formatargspec(args, varargs, varkw)

        # Figure out which arguments to use in the key, either the names
        # provided in the constructor, or all args if none were given.
        if self.arg_names:
            use_args = self.arg_names
        else:
            use_args = set(args)
            if varargs: use_args.add(varargs)
            if varkw: use_args.add(varkw)

        # Get the kwargs_lines filler, which is either empty, or creates
        # kwargs_tuple from kwargs.
        filler['kwargs_lines'] = ''
        if varkw and varkw in use_args:
            filler['kwargs_lines'] = (
                '    %(varkw)s_list = %(varkw)s.items()\n'
                '    %(varkw)s_list.sort()\n'
                '    %(varkw)s_tuple = tuple(%(varkw)s_list)\n\n'
                % dict(varkw=varkw)
            )

        # Figure out the 'key' item, which is a little tricky.
        key_items = []

        args_items = [arg for arg in args if arg in use_args]
        if len(args_items) == 1:
            key_items.append('(%s,)' % args_items[0])
        else:
            key_items.append('(%s)' % ', '.join(args_items))

        if varargs in use_args:
            key_items.append(varargs)

        if varkw in use_args:
            key_items.append('%s_tuple' % varkw)

        # Special case: key_items contains only ['(one_item)']. In this case
        # the key doesn't need to be a tuple.
        if len(key_items) == 1 and len(args_items) == 1:
            filler['key'] = args_items[0]
        else:
            filler['key'] = ' + '.join(key_items)

        return (filler['name'], func_template % filler)

class CacheDebugDecorator(CacheIntroDecorator):
    def cache_factory(self):
        return DebugCache()

class CacheNullDecorator(CacheDecorator):
    def __call__(self, func):
        return func

# These are specific decorators. Typically they should not be used directly,
# but in a pinch they can be. Normally the 'cacheit*' decorators should be used,
# which are set based on the cache options.
#
# Decorators used without arguments:
cache_decorator = CacheIntroDecorator()
cache_null_decorator = CacheNullDecorator()
cache_debug_decorator = CacheDebugDecorator()
# Decorators used with arguments:
cache_args_decorator = CacheIntroDecorator
cache_null_args_decorator = CacheNullDecorator

def print_cache(file=None):
    """print cache content"""

    return registry.print_cache(file)

def clear_cache():
    """clear cache content"""

    return registry.clear()

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
    cacheit_args        = cache_null_args_decorator
elif _usecache in ('yes', 'debug'):
    cacheit      = cache_decorator
    cacheit_args = cache_args_decorator
else:
    raise RuntimeError('unknown argument in SYMPY_USE_CACHE: %s' % _usecache)
