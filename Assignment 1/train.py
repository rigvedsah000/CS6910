import sgd,mgd

def train(train_x, train_y, d, hl, ol,batch_size = 16):

    print("Function Invoked: train")

    # return sgd.sgd(train_x, train_y, d, hl, ol)

    return mgd.mgd(train_x,train_y,d,hl,ol,batch_size)