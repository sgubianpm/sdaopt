import pytest

@pytest.mark.skip(reason="Need rpy2 installed")
def test_pyfun2r_ops_inc():
    from benchmark import convert2r
    code = 'x = x[0]'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x[1]')


@pytest.mark.skip(reason="Need rpy2 installed")
def test_pyfun2r_ops_plusminus():
    from benchmark import convert2r
    code = 'x += 100'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x+100')

@pytest.mark.skip(reason="Need rpy2 installed")
def test_pyfun2r_ops_power():
    from benchmark import convert2r
    code = 'x = x ** (x+1)'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-x^(x+1)')

@pytest.mark.skip(reason="Need rpy2 installed")
def test_pyfun2r_ops_self():
    from benchmark import convert2r
    code = 'x = self.N / x[0]'
    rcode = convert2r.pyfun2r(code)
    assert(rcode.replace(' ', '') == 'x<-N/x[1]')





