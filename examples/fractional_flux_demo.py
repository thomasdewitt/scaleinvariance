from scaleinvariance.simulation import fractionally_integrated_flux

try:
    fif_data = fractionally_integrated_flux(2048, 0.3)
    print("FIF generated successfully")
except NotImplementedError:
    print("FIF simulation not yet implemented")