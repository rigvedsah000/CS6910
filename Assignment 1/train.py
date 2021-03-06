import sgd,mgd,nag

def train(train_x, train_y, d, hl, ol,batch_size):

    print("Function Invoked: train")

    # return sgd.sgd(train_x, train_y, d, hl, ol)

    # return mgd.mgd(train_x,train_y,d,hl,ol,batch_size)

    return nag.nag(train_x,train_y,d,hl,ol,batch_size)