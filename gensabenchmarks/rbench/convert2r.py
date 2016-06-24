import inspect
from collections import OrderedDict
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP, NEWLINE
from io import BytesIO

import rpy2.robjects
from rpy2.robjects.vectors import ListVector

def pyfun2r(code, add_context=False):
    result = []
    g = tokenize(BytesIO(code.encode('utf-8')).readline)
    context = 'EXPR'
    previous = None
    for toknum, tokval, _, _, cline  in g:
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
                result.append((toknum, value))
        elif toknum == OP:
            if tokval == '.' and context == 'SELF':
                context = 'EXPR'
                continue
            if tokval == '**':
                result.append((OP, '^'))
            elif tokval == '=':
                result.append((OP, '<-'))
            elif tokval == '+=':
                result.extend([
                    (OP, '<-'),
                    previous,
                    (OP, '+')
                    ])
            elif tokval == '[':
                context = 'INDEX'
                result.append((OP, '['))
            elif tokval == ']':
                context = 'EXPR'
                result.append((OP, ']'))
            else:
                result.append((OP, tokval))
        elif toknum == NUMBER:
            if context == 'INDEX':
                v = int(str(tokval)) + 1
                result.append((NUMBER, str(v)))
            else:
                result.append((NUMBER, tokval))
        else:
            result.append((toknum, tokval))
        previous = (toknum, tokval)
    if add_context:
        # Adding end of function character
        result.append((OP, '}'))
    return untokenize(result).decode('utf-8')


class GOClass2RConverter(object):
    def __init__(self, klass):
        self._klass = klass
        self._instance = klass()
        self.r = rpy2.robjects.r
        self.R = rpy2.robjects

    @property
    def name(self):
        return type(self._instance).__name__

    @property
    def lowerb(self):
        lower = [x[0] for x in self._instance.bounds]
        return self.R.FloatVector(lower)

    @property
    def upperb(self):
        upper = [x[0] for x in self._instance.bounds]
        return self.R.FloatVector(upper)

    @property
    def xglob(self):
        go = self._instance.global_optimum
        return self.R.FloatVector(go)

    @property
    def fglob(self):
        fglob = self._instance.fglob
        return fglob

    @property
    def fun(self):
        fun_code = inspect.getsourcelines(self._instance.fun)[0]
        # Remove first line which contains func definition
        del fun_code[0]
        fun_code = '\n'.join(fun_code)
        r_fun_code = '{0} <- function(x) {{ {1}) }}'.format(
                'fun', pyfun2r(fun_code))
        r_fun = self.r(r_fun_code)
        return r_fun

    @property
    def rlist(self):
        od = OrderedDict()
        od.update((('lowerb', self.lowerb),))
        od.update((('upperb', self.upperb),))
        od.update((('xglob', self.xglobal),))
        od.update((('fglob', self.fglob),))
        od.update((('fun', self.fun),))
        return ListVector(od)


