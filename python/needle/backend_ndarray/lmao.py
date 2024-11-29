val = ops.split(X, axis=0) 
h_n = []
if h0:
    h0 = ops.split(h0, axis=0)
for layer in range(self.num_layers):
    if h0:
        val1 = ops.tuple_get_item(h0, layer)
    else:
        val1 = None
    H_l = [] 
    for i in range(X.shape[0]):
        val1 = self.rnn_cells[layer](ops.tuple_get_item(val, i), val1) 
        H_l.append(val1)
    val = ops.make_tuple(*H_l)
    h_n.append(val1)          
return ops.stack(val, axis=0), ops.stack(h_n, axis=0)