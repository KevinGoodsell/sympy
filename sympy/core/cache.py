""" Caching facility for SymPy """

# TODO:
# * Add full documentation and tests
# * Add a global function to create a cache

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

class OverrideCache(Cache):
    '''Cache designed to allow easy overriding of item setting and retrieval.
    This is just to make custom caches easy (e.g., for debugging) but is not
    designed to be fast.

    To customize, override _set_item and _get_item.'''

    def __init__(self, seq=(), **kwargs):
        Cache.__init__(self)
        self.update(seq, **kwargs)

    def _set_item(self, key, value):
        Cache.__setitem__(self, key, value)

    def _get_item(self, key):
        return Cache.__getitem__(self, key)

    def __setitem__(self, key, value):
        self._set_item(key, value)

    def __getitem__(self, key):
        return self._get_item(key)

    def get(self, key, default=None):
        try:
            return self._get_item(key)
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        try:
            return self._get_item(key)
        except KeyError:
            self._set_item(key, default)
            return default

    def update(self, seq=(), **kwargs):
        if hasattr(seq, 'iteritems'):
            it = seq.iteritems()
        else:
            it = iter(seq)

        for (key, value) in it:
            self._set_item(key, value)

        for (key, value) in kwargs:
            self._set_item(key, value)

class NullCache(Cache):
    '''Cache that never stores anything. To keep overhead minimal, this
    does not use OverrideCache.'''

    def __init__(self, seq=(), **kwargs):
        Cache.__init__(self)

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

class DebugCache(OverrideCache):
    '''Cache that never produces anything, but verifies all added items
    to ensure they are immutable and match previously added items with
    the same key.'''

    def __init__(self, seq=(), **kwargs):
        # Items are stored in a separate dict so that nothing ever comes
        # out of the cache.
        self._cache = {}
        OverrideCache.__init__(self, seq, **kwargs)

    def _set_item(self, key, value):
        cached = self._cache.setdefault(key, value)
        assert cached == value, 'cached value != new value'

        # Verify the immutability of value. Hashability roughly
        # corresponds to immutability.
        hash(value)

class StatsCache(OverrideCache):
    def __init__(self, seq=(), **kwargs):
        self.attempts = 0
        self.misses = 0
        OverrideCache.__init__(self, seq, **kwargs)

    def _get_item(self, key):
        self.attempts += 1
        try:
            return OverrideCache._get_item(self, key)
        except KeyError:
            self.misses += 1
            raise

def _cache_factory():
    if _usecache == 'no':
        return NullCache()
    elif _usecache == 'debug':
        return DebugCache()
    elif _usecache == 'stats':
        return StatsCache()
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

    def iter_caches(self):
        for (key, (cache, comment)) in self._registry.iteritems():
            yield (key, cache, comment)

    def clear(self):
        '''Clear all items from all caches.'''
        for (cache, comment) in self._registry.values():
            cache.clear()

    def print_cache(self, file=None, verbose=False):
        '''Print all caches to file, which defaults to sys.stdout. Intended
        for debugging only.'''
        if file is None:
            import sys
            file = sys.stdout

        for (ckey, cache, comment) in self.iter_caches():
            if not cache and not verbose:
                continue

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

            if text:
                lines = ['%-*s : %s' % (width, k, v) for (k, v) in text]
                print >> file, '\n'.join(lines)
            else:
                print >> file, 'empty'

registry = CacheRegistry()

def print_cache(file=None, verbose=False):
    """print cache content"""

    return registry.print_cache(file, verbose)

def clear_cache():
    """clear cache content"""

    return registry.clear()

def print_stats(file=None, verbose=False):
    if file is None:
        import sys
        file = sys.stdout

    for (key, cache, comment) in registry.iter_caches():
        if not isinstance(cache, StatsCache):
            continue

        ident = comment or key
        if cache.attempts:
            hits = cache.attempts - cache.misses
            percent = 100.0 * hits / cache.attempts
            print >> file, '%s: %d misses, %d hits, %.1f%%' % (ident,
                cache.misses, hits, percent)
        elif verbose:
            print >> file, '%s: unused' % ident

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

class CacheStatsDecorator(CacheIntroDecorator):
    def cache_factory(self):
        return StatsCache()

# These are specific decorators. Typically they should not be used directly,
# but in a pinch they can be. Normally the 'cacheit*' decorators should be used,
# which are set based on the cache options.
#
# Decorators used without arguments:
cache_decorator = CacheIntroDecorator()
cache_null_decorator = CacheNullDecorator()
cache_debug_decorator = CacheDebugDecorator()
cache_stats_decorator = CacheStatsDecorator()
# Decorators used with arguments:
cache_args_decorator = CacheIntroDecorator
cache_null_args_decorator = CacheNullDecorator


# SYMPY_USE_CACHE=yes/no/debug/stats
import os
_usecache = os.getenv('SYMPY_USE_CACHE', 'yes').lower()

if _usecache == 'no':
    cacheit             = cache_null_decorator
    cacheit_args        = cache_null_args_decorator
elif _usecache in ('yes', 'debug', 'stats'):
    cacheit      = cache_decorator
    cacheit_args = cache_args_decorator
    if _usecache == 'stats':
        import atexit
        atexit.register(print_stats)
else:
    raise RuntimeError('unknown argument in SYMPY_USE_CACHE: %s' % _usecache)
