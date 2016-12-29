##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 -*-
import inspect
from collections import OrderedDict
from tokenize import untokenize, NUMBER, NAME, OP, NEWLINE
from tokenize import generate_tokens
from io import BytesIO

import rpy2.robjects
from rpy2.robjects.vectors import ListVector


def pyfun2r(code, add_context=False):
    result = []
    g = generate_tokens(BytesIO(code.encode('utf-8')).readline)
    context = 'EXPR'
    previous = None
    for toknum, tokval, _, _, cline in g:
        if toknum == NEWLINE:
            continue
        if toknum == NAME:
            value = tokval
            if tokval == 'return':
                context = 'RETURN'
                result.extend([
                    (NAME, 'RET'),
                    (OP, '<- ')
                    ])
            else:
                if tokval == 'self':
                    context = 'SELF'
                    continue
                if tokval == 'arange':
                    value = 'seq'
                elif tokval == 'nfev':
                    context = 'INC'
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
                if context == 'INC':
                    result.extend([
                        (OP, '<<-'),
                        previous,
                        (OP, '+')
                    ])
                    context = 'EXPR'
                else:
                    result.extend([
                        (OP, '<-'),
                        previous,
                        (OP, '+')
                        ])
            elif tokval == '/=':
                result.extend([
                    (OP, '<-'),
                    previous,
                    (OP, '/')
                    ])
            elif tokval == '-=':
                result.extend([
                    (OP, '<-'),
                    previous,
                    (OP, '-')
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
    def dim(self):
        return self._instance.N

    @property
    def lowerb(self):
        lower = [x[0] for x in self._instance.bounds]
        return self.R.FloatVector(lower)

    @property
    def upperb(self):
        upper = [x[1] for x in self._instance.bounds]
        return self.R.FloatVector(upper)

    @property
    def xglob(self):
        go = self._instance.global_optimum
        return self.R.FloatVector(list(go[0]))

    @property
    def fglob(self):
        fglob = self._instance.fglob
        return fglob

    @property
    def fun(self):
        fun_code = inspect.getsourcelines(self._instance.fun)[0]
        # Remove 2 first lines which contains func definition and
        # counter increment
        del fun_code[0:1]
        fun_code = '\n'.join(fun_code)
        r_fun_code = '{0} <- function(x) {{ N <- {1}; {2}; if (is.na(RET)) RET <- 1e13; if (firstHit && RET <= TolF) {{ fn.call.suc <<- nfev; feval.suc <<- RET; firstHit <<- FALSE;}}\nreturn(RET) }}'.format(
                'fun', self.dim, pyfun2r(fun_code))
        r_fun = self.r(r_fun_code)
        return r_fun

    @property
    def fun_no_context(self):
        fun_code = inspect.getsourcelines(self._instance.fun)[0]
        # Remove 2 first lines which contains func definition and
        # counter increment
        del fun_code[0:1]
        fun_code = '\n'.join(fun_code)
        r_fun_code = '{0} <- function(x) {{ N <- {1}; nfev <- 0; {2}; if (is.na(RET)) RET <- 1e13; \nreturn(RET) }}'.format(
                'fun', self.dim, pyfun2r(fun_code))
        r_fun = self.r(r_fun_code)
        return r_fun

    @property
    def rlist(self):
        od = OrderedDict()
        od.update((('name', self.name),))
        od.update((('dim', self.dim),))
        od.update((('lower', self.lowerb),))
        od.update((('upper', self.upperb),))
        od.update((('xglob', self.xglob),))
        od.update((('glob.min', self.fglob),))
        od.update((('fn', self.fun),))
        return ListVector(od)
