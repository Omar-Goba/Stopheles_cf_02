### importations ###


def init_tensor(arr: np.array) -> tuple[np.array, np.array]:
    """
    """
    ...

def main():
    """
    """
    ### load data ###
    tr = pd.read_csv("./dbs/cooked/tr.csv")
    ts = pd.read_csv("./dbs/cooked/ts.csv")
    
    ### convert to numpy ###
    tr = tr.to_numpy()
    ts = ts.to_numpy()

    ### init tensors ###
    tr_X, tr_y = init_tensor(tr)
    ts_X, ts_y = init_tensor(ts)

    ### save tensors in `dbs/cooked/` ###

if (__name__ == "__main__"):    main()
