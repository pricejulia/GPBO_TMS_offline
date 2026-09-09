import Tools.helper as h
import json


class Config:

    def __init__(self,config_path,overwrite_save_folder=None):

        self.expected_attributes = ['exp_type',
                                    'GP_library',
                                    'torch_device',
                                    'search_space',
                                    'random_pts',
                                    'kappa',
                                    'rho_min',
                                    'rho_max',
                                    'noise_min',
                                    'noise_max',
                                    'max_it',
                                    'repeats',
                                    'offline',
                                    'NUM_TRIALS',
                                    'debug_mode',
                                    'verbose',
                                    'maximization',
                                    'mep_window',
                                    'zero_sample',
                                    'baseline_window',]
        
        self.config_identifier = config_path.split('config')[1].split('.json')[0]
        
        with open(config_path, 'r') as file:
            config_data = json.load(file)

        # make sure no missing attribute
        self.passed_attributes = [*config_data]

        for i in self.expected_attributes:
            if i not in self.passed_attributes:
                print(f'Expected attribute "{i}" in the config.json file but it is missing! \n')
            
        for key, value in config_data.items():

            if key in self.expected_attributes:
                setattr(self,key,value)
            else:
                print(f'Got unexpected attribute {key}')
                
        self.save_folder = overwrite_save_folder

        if not self.debug_mode and self.save_folder is None:
            self.create_log_folder()

    def create_log_folder(self):
        if not self.debug_mode:
            self.save_folder = h.setup_folder('Results/Learning_curves')

    def log_config(self):

        del self.expected_attributes
        del self.passed_attributes

        obj_dict = vars(self)

        json_data = json.dumps(obj_dict, indent=1)

        if not self.debug_mode:
            with open(self.save_folder+f"/log_config{self.config_identifier}.json", "w") as json_file:
                json_file.write(json_data)