import numpy as np, os
for ds in ['all', 'men', 'women']:
    base = 'data/processed_v2/' + ds
    if not os.path.exists(base): continue
    train = np.load(base + '/context_train.npz')
    val = np.load(base + '/context_val.npz')
    test = np.load(base + '/context_test.npz')
    print('OBD-' + ds)
    print('  Train:', train['contexts'].shape, 'clicks:', int(train['click'].sum()), 'rate:', float(train['click'].mean()))
    print('  Val:  ', val['contexts'].shape, 'clicks:', int(val['click'].sum()), 'rate:', float(val['click'].mean()))
    print('  Test: ', test['contexts'].shape, 'clicks:', int(test['click'].sum()), 'rate:', float(test['click'].mean()))
if os.path.exists('data/processed_criteo'):
    base = 'data/processed_criteo'
    train = np.load(base + '/context_train.npz')
    val = np.load(base + '/context_val.npz')
    test = np.load(base + '/context_test.npz')
    print('Criteo')
    print('  Train:', train['contexts'].shape, 'convs:', int(train['conversion'].sum()), 'rate:', float(train['conversion'].mean()))
    print('  Val:  ', val['contexts'].shape, 'convs:', int(val['conversion'].sum()), 'rate:', float(val['conversion'].mean()))
    print('  Test: ', test['contexts'].shape, 'convs:', int(test['conversion'].sum()), 'rate:', float(test['conversion'].mean()))
