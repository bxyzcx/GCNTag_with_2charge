import argparse
import configparser


def init_args(file):
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    config.read(file)
    # add train params
    parser.add_argument("--engine_model", default=config["train"]["engine_model"], type=int)
    parser.add_argument("--train_dir", default=config["train"]["train_dir"], type=str)
    parser.add_argument("--num_workers", default=config["train"]["num_workers"], type=int)
    parser.add_argument("--batch_size", default=config["train"]["batch_size"], type=int)
    parser.add_argument("--num_epoch", default=config["train"]["num_epoch"], type=int)
    parser.add_argument("--init_lr", default=config["train"]["init_lr"], type=float)
    parser.add_argument("--steps_per_validation", default=config["train"]["steps_per_validation"], type=int)
    parser.add_argument("--weight_decay", default=config["train"]["weight_decay"], type=float)
    parser.add_argument("--MAX_NUM_PEAK", default=config["train"]["MAX_NUM_PEAK"], type=int)
    parser.add_argument("--MZ_MAX", default=config["train"]["MZ_MAX"], type=float)
    parser.add_argument("--MAX_LEN", default=config["train"]["MAX_LEN"], type=int)
    parser.add_argument("--num_ion", default=config["train"]["num_ion"], type=int)
    parser.add_argument("--TOL", default=config["train"]["TOL"], type=float)
    parser.add_argument("--IF_PreProcess", default=config["train"]["IF_PreProcess"], type=int)
    parser.add_argument("--MAX_TAG_NUM", default=config["train"]["MAX_TAG_NUM"], type=int)

    # add model params
    parser.add_argument("--input_dim", default=config["model"]["input_dim"], type=int)
    parser.add_argument("--output_dim", default=config["model"]["output_dim"], type=int)
    parser.add_argument("--n_classes", default=config["model"]["n_classes"], type=int)
    parser.add_argument("--edges_classes", default=config["model"]["edges_classes"], type=int)
    parser.add_argument("--units", default=config["model"]["units"], type=int)
    parser.add_argument("--use_lstm", default=False, type=bool)
    parser.add_argument("--lstm_hidden_units", default=config["lstm"]["lstm_hidden_units"], type=int)
    parser.add_argument("--embedding_size", default=config["lstm"]["embedding_size"], type=int)
    parser.add_argument("--num_lstm_layers", default=config["lstm"]["num_lstm_layers"], type=int)
    parser.add_argument("--dropout", default=config["lstm"]["dropout"], type=float)

    parser.add_argument("--beam_size", default=config["search"]["beam_size"], type=int)
    parser.add_argument("--knapsack", default=config["search"]["knapsack"], type=str)

    # add data params
    parser.add_argument("--input_spectrum_file_train", default=config["data"]["input_spectrum_file_train"], type=str)
    parser.add_argument("--input_feature_file_train", default=config["data"]["input_feature_file_train"], type=str)
    parser.add_argument("--input_spectrum_file_valid", default=config["data"]["input_spectrum_file_valid"], type=str)
    parser.add_argument("--input_feature_file_valid", default=config["data"]["input_feature_file_valid"], type=str)

    parser.add_argument("--denovo_input_spectrum_file", default=config["data"]["denovo_input_spectrum_file"], type=str)
    parser.add_argument("--denovo_input_feature_file", default=config["data"]["denovo_input_feature_file"], type=str)
    parser.add_argument("--denovo_output_file", default=config["data"]["denovo_output_file"], type=str)
    parser.add_argument("--accuracy_file", default=config["data"]["denovo_output_file"] + "_model", type=str)
    parser.add_argument("--denovo_only_file", default=config["data"]["denovo_output_file"] + ".denovo_only", type=str)
    parser.add_argument("--scan2fea_file", default=config["data"]["denovo_output_file"] + ".scan2fea", type=str)
    parser.add_argument("--multifea_file", default=config["data"]["denovo_output_file"] + ".multifea", type=str)


    args = parser.parse_args()

    return args



