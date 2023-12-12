# eeg-decoding

## Dataset
The ZuCo 1.0 dataset is available at: https://osf.io/q3zws/ <br />
The ZuCo 2.0 dataset is available at: https://osf.io/2urht/

## Getting Started

### Data Preparation
1. **Download EEG Data**: 
   - Obtain the EEG data from the OSF directory.
   - Store the data in `dataset/zuco1` or `dataset/zuco2`.

### Configuration
2. **Set Up for EEG Extraction**:
   - In `config.py`, enable EEG extraction by setting `run_eeg_extraction` to `True`.
   - Specify `feature_set` and `class_task` according to your requirements.
   - Ensure the directory `../eeg_features/` is created to hold the extracted EEG features.

### Running Experiments
3. **Experiment Execution**:
   - For running experiments, set `run_eeg_extraction` to `False`.
   - Adjust other parameters as needed for your specific experiment.
   - Execute `python3 tune_text_model.py` for the text-only baseline model.
   - Use `python3 tune_combi_model.py` to run other models.

## Notes
- Ensure all necessary libraries and dependencies are installed prior to running the experiments.
- For detailed information on parameter settings and model configurations, refer to the documentation in `config.py`.

## Contributions
Contributions to this project are welcome. Please follow the guidelines outlined in `CONTRIBUTING.md`.

## License
This project is licensed under the [LICENSE NAME] - see the `LICENSE.md` file for details.

## Acknowledgments
- Mention any collaborators, data sources, or special acknowledgments here.

## Contact
For queries or further information, please contact [Your Contact Information].
