from model import epi_model


if __name__ == '__main__':
    # Put parameters in a Pythonic dictionary.
    dim_size = 1000
    param_values = {"beta": 0.20, "rho": 0.3, "life_time": 11, 'run_time' : 501, 'dim': [dim_size, dim_size],
                    "epi_c": (100, 50), 'lattice': 'l_hill', 'data_set_name':'dat__.npy'}

    sim_name = 'beta_' + str(param_values['beta']).strip('0.') + '_rho_' + str(param_values['rho']).strip('0.')
    epi_model.main(param_values, sim_name)
    print('success')
