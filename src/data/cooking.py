### importations ###


def init_tensor(arr: np.array) -> tuple[np.array, np.array]:
    """
    """
    ...

def main():
    """
    """
    tr = pd.read_csv("./dbs/cooked/tr.csv")
    ts = pd.read_csv("./dbs/cooked/ts.csv")
    
    tr = tr.to_numpy()
    ts = ts.to_numpy()

    tr_X, tr_y = init_tensor(tr)
    ts_X, ts_y = init_tensor(ts)

if (__name__ == "__main__"):    main()
