import numpy as np


def select_amb(data, ambsel):
    """
    Author: Richard Lindsley (lindsley@remss.com)
    Select the UHR ambiguity to use.

    data: shaped as (rows, cols, ambs), ambs is 4
    ambsel: (rows, cols), indexes data; ranges from 1-4, 0 is no-data

    Return a masked array.

    """
    # Note that ambsel goes from 1 to 4 (1-based indexing),
    # but -1 and 0 both flag bad values.
    # ambsel = ambsel.copy()
    # ambsel[ambsel == -1] = 0
    # ambsel = 1
    # data_opts = [np.nan, data[:, :, 0], data[:, :, 1],
    #              data[:, :, 2], data[:, :, 3]]
    # data_sel = ma.choose(ambsel, data_opts)

    # This version is a little faster
    data_sel = np.zeros(data[:, :, 0].shape)
    mask_bad = (ambsel == -1) | (ambsel == 0)
    mask_0 = (ambsel == 1)
    mask_1 = (ambsel == 2)
    mask_2 = (ambsel == 3)
    mask_3 = (ambsel == 4)
    data_sel[mask_bad] = np.nan
    data_sel[mask_0] = data[:, :, 0][mask_0]
    data_sel[mask_1] = data[:, :, 1][mask_1]
    data_sel[mask_2] = data[:, :, 2][mask_2]
    data_sel[mask_3] = data[:, :, 3][mask_3]

    return np.ma.masked_invalid(data_sel)
