import rpy2.robjects
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
from io import BytesIO
import go_benchmark_functions as gbf

def goclass():
    """
    Generator to get global optimization test classes/functions
    defined in SciPy
    """
    benchmark_functions = [item for item in bench_members if
            issubclass(item[1], gbf.Benchmark)]
    for name, klass in self.benchmark_functions:
        yield klass

def pyfun2r(code):
    result = []
    g = tokenize(BytesIO(code.encode('utf-8')).readline)
    context = 'EXPR'
    previous = None
    for toknum, tokval, _, _, cline  in g:
        # Ignoring function definition
        if cline.strip() == 'def fun(self, x, *args):':
            continue
        # Ignoring self counter line
        if cline.strip() == 'self.nfev += 1':
            continue
        if toknum == NEWLINE:
            continue
        if toknum == NAME:
            value = tokval
            if tokval == 'return':
                context = 'RETURN'
                result.extend([
                    (NAME, tokval),
                    (OP, '(')
                    ])
            else:
                if tokval == 'self':
                    context = 'SELF'
                    continue
                if tokval == 'arange':
                    value = 'seq'
                result.append(toknum, value)
        elif toknum == OP:
            if tokval == '.' and context == 'SELF':
                context = 'EXPR'
                continue
            if tokval == '**':
                result.append((OP, '^'))
            if tokval == '=':
                result.append((OP, '<-'))
            if tokval == '+=':
                result.extend([
                    (OP, '<-'),
                    previous,
                    (OP, '+')
                    ])
        previous = (toknum, tokval)








class GOClass2RConverter(object):
    def __init__(self, klass):
        self._klass = klass
        self._instance = klass()
        self.r = rpy2.robjects.r
        self.R = rpy2.robjects

    @property
    def name(self):
        return type(self).__name__

    @property
    def bounds(self):
        bounds = self._instance._bounds
        return self.R.FloatVector(bounds)

    @property
    def x_global(self):
        go = self._instance.global_optimum
        return self.R.FloatVector(go)

    @property
    def fglob(self):
        fglob = self._instance.fglob
        return fglob

    @property
    def func(self):
        fun_code = inspect.getsourcelines(self._instance().fun)[0]
        fun_code = '\n'.join(fun_code)
        r_fun_code = pyfun2r(fun_code)

    @property
    def rlist(self):
        pass


