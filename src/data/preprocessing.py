
def main() -> int:
    """
        Preprocesses the raw data into a format that is easier to work with.
        args:
            None
        returns:
            0 if successful
    """
    ### load data ###
    root_dir = "./"
    with open(f"{root_dir}dbs/raw/db.csv", "r") as f:
        txt = f.read()

    ### format data ###
    txt = txt.replace("\n~~~", "")

    ### turn into list of lists ###
    df = txt.split("\n")
    df = [row.split(",") for row in df]

    ### assert ###
    assert all([len(row) == 4 for row in df]), "Error: Invalid row length"

    ### save data ###
    with open(f"{root_dir}dbs/intermittent/db.csv", "w") as f:
        f.write(txt)

    return 0

if (__name__ == "__main__"):    main()
#   __   _,_ /_ __, 
# _(_/__(_/_/_)(_/(_ 
#  _/_              
# (/                 
