import numpy as np
import pandas as pd
import warnings


def interpolate_experimental_data(ud_array, voltage_exp, current_exp):

    max_voltage_exp = np.max(np.abs(voltage_exp))
    voltage_exp /= max_voltage_exp
    voltage_exp_np = voltage_exp.to_numpy(dtype=np.float64)
    current_exp_np = current_exp.to_numpy(dtype=np.float64)

    max_voltage = 0.99999
    max_voltage_index_ud = np.argmax(ud_array > max_voltage)
    max_voltage_index_exp = np.argmax(voltage_exp_np > max_voltage)
    new_current_first = np.interp(
        ud_array[:max_voltage_index_ud],
        voltage_exp_np[:max_voltage_index_exp],
        current_exp_np[:max_voltage_index_exp],
    )

    # Flip it because np.interp expects increasing x argument
    new_current_second = np.flip(
        np.interp(
            np.flip(ud_array[max_voltage_index_ud:]),
            np.flip(voltage_exp_np[max_voltage_index_exp:]),
            np.flip(current_exp_np[max_voltage_index_exp:]),
        )
    )

    return np.concatenate((new_current_first, new_current_second))


def obtain_experiment_cv_data(scan_rate):
    xls = pd.ExcelFile("./coin_cell_data.xlsx")
    cv_data = pd.read_excel(xls, "CV")
    working_area = 1.0
    mass_loading = 1.0

    def clean_np_arr(arr):
        arr = arr[1::]
        return arr[~np.isnan(arr.to_numpy(dtype=float))]

    unit_current = (1e-3 / working_area) / (
        mass_loading / 1000
    )  # mA/A , 1/cm^2, mg/cm^2
    columns = [column for column in cv_data.columns]
    column_title = f"{scan_rate:.0f} mV/s"
    if column_title in columns:
        voltage = cv_data[column_title]
        index_current = columns.index(column_title) + 1
        current_density = cv_data[columns[index_current]]

        current_density = clean_np_arr(current_density) * unit_current
        voltage = clean_np_arr(voltage)
        return voltage, current_density
    else:
        warnings.warn(f"No data found for: {scan_rate}")


def obtain_experiment_cp_data(current):
    xls = pd.ExcelFile("./experimental_data.xlsx")
    cp_data = pd.read_excel(xls, "CP")

    def clean_np_arr(arr):
        arr = arr[1::]
        return arr[~np.isnan(arr.to_numpy(dtype=float))]

    columns = [column for column in cp_data.columns]
    column_title = f"non 3D GA {current:g} A/g"
    if column_title in columns:
        time = cp_data[column_title]
        index_voltage = columns.index(column_title) + 1
        voltage = cp_data[columns[index_voltage]]

        time = clean_np_arr(time)
        voltage = clean_np_arr(voltage)
        return time, voltage
    else:
        warnings.warn(f"No data found for: {current}")
