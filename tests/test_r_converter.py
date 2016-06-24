from gensabenchmarks.rbench import convert2r

def test_pyfun2r_ops_inc():
    code = 'x = x[0]'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x[1]')


def test_pyfun2r_ops_plusminus():
    code = 'x += 100'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x+100')

def test_pyfun2r_ops_power():
    code = 'x = x ** (x+1)'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x^(x+1)')

def test_pyfun2r_ops_self():
    code = 'x = self.N / x[0]'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-N/x[1]')





