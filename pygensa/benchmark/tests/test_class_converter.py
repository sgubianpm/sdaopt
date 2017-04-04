import pytest
import numpy.testing as npt

@pytest.mark.skip(reason="Need rpy2 installed")
def test_convert_fun0():
    from benchmark.convert2r import GOClass2RConverter
    gocc = GOClass2RConverter(go_funcs_A.Ackley01)
    assert(gocc.name == 'Ackley01')
    assert(gocc.fglob == 0)
    assert(gocc.dim == 2)

@pytest.mark.skip(reason="Need rpy2 installed")
def test_convert_fun0_rlist():
    from benchmark.convert2r import GOClass2RConverter
    gocc = GOClass2RConverter(go_funcs_A.Ackley01)
    rl = gocc.rlist
    idx_fglob = rl.names.index('glob.min')
    assert(idx_fglob)
    obj = rl[idx_fglob][0]
    assert(obj == 0.)

@pytest.mark.skip(reason="Need rpy2 installed")
def test_convert_run_rfun():
    from benchmark.convert2r import GOClass2RConverter
    gocc = GOClass2RConverter(go_funcs_A.Ackley01)
    rfun = gocc.fun
    rxglob = gocc.xglob
    npt.assert_almost_equal(gocc.fglob, rfun(rxglob)[0])

@pytest.mark.skip(reason="Need rpy2 installed")
def test_convert_func_calling():
    from benchmark.convert2r import GOClass2RConverter
    count = 0
    notok = 0
    for name, klass in goclass():
        print('Calling function: {0}'.format(name))
        count += 1
        try:
            gocc = GOClass2RConverter(klass)
            #with nostdout():
            res = gocc.fun(gocc.xglob)
            npt.assert_almost_equal(gocc.fglob, res[0], decimal=4)
        except Exception as e:
            print(e)
            notok += 1
            continue
    print("R func call that failed: {0} ratio: {1}".format(notok, (count-notok)*100/count))




