from gensabenchmarks.go_benchmark_functions import go_funcs_A
from gensabenchmarks.rbench.convert2r import GOClass2RConverter

def test_convert_fun0():
    gocc = GOClass2RConverter(go_funcs_A.Ackley01)
    assert(gocc.name == 'Ackley01')
    assert(gocc.fglob == 0)

def test_convert_fun0_rlist():
    gocc = GOClass2RConverter(go_funcs_A.Ackley01)
    rl = gocc.rlist
    idx_fglob = rl.names.index('fglob')
    assert(idx_fglob)
    obj = rl[idx_fglob]
    assert(obj == 0.)



