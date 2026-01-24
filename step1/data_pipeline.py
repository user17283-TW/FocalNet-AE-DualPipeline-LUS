import pandas as pd
import os

def load_data(data_path: str) -> pd.DataFrame:

    data = pd.read_csv(data_path)

    required_labels_data = data[
        data["Label"].isin(["Bacterial pneumonia", "regular", "COVID-19"])
    ]

    num_total_bacterial = len(
        required_labels_data[required_labels_data["Label"] == "Bacterial pneumonia"]
    )
    num_total_regular = len(
        required_labels_data[required_labels_data["Label"] == "regular"]
    )
    num_total_covid19 = len(
        required_labels_data[required_labels_data["Label"] == "COVID-19"]
    )

    num_train_bacterial = int(num_total_bacterial * 0.7)
    num_val_bacterial = int(num_total_bacterial * 0.1)

    num_train_regular = int(num_total_regular * 0.7)
    num_val_regular = int(num_total_regular * 0.1)

    num_test_covid19 = int(num_total_covid19 * 0.5)

    data.loc[data["Label"] == "Bacterial pneumonia", "Use"] = [
        (
            "Train"
            if i < num_train_bacterial
            else "Val" if i < num_train_bacterial + num_val_bacterial else "Test"
        )
        for i in range(num_total_bacterial)
    ]
    data.loc[data["Label"] == "regular", "Use"] = [
        (
            "Train"
            if i < num_train_regular
            else "Val" if i < num_train_regular + num_val_regular else "Test"
        )
        for i in range(num_total_regular)
    ]
    data.loc[data["Label"] == "COVID-19", "Use"] = [
        "Test" if i < num_test_covid19 else "Val" for i in range(num_total_covid19)
    ]

    modified_data_path_with_covid = "processed_data_modified.csv"
    data.to_csv(modified_data_path_with_covid, index=False)

    print(modified_data_path_with_covid)


def randomly_load(
    data_path: str, split_ratio: tuple = (0.5, 0.1, 0.4), covid_ratio: float = 0.5, use_other=False, output_path="temporary_data.csv"
) -> dict:

    assert (
        sum(split_ratio) == 1
    ), f"The sum of the split ratio must be equal to 1 , Got {sum(split_ratio)}"

    data = pd.read_csv(data_path)
    regular_df = data[data["Label"] == "regular"]
    pneumonia_df = data[(data["Label"] == "Bacterial pneumonia") | (data["Label"] == "Pneumonia")]
    covid_df = data[data["Label"] == "COVID-19"]

    # Training set contains 50% of Bacterial pneumonia and regular images
    # Validation set contains 10% of Bacterial pneumonia and regular images
    # Test set contains 40% of Bacterial pneumonia and regular images and 50% of COVID-19 images
    bacterial_training = pneumonia_df.sample(frac=split_ratio[0])
    pneumonia_df = pneumonia_df.drop(bacterial_training.index)
    bacterial_validation = pneumonia_df.sample(
        frac=split_ratio[1] / (1 - split_ratio[0])
    )
    pneumonia_df = pneumonia_df.drop(bacterial_validation.index)
    bacterial_test = pneumonia_df

    regular_training = regular_df.sample(frac=split_ratio[0])
    regular_df = regular_df.drop(regular_training.index)
    regular_validation = regular_df.sample(frac=split_ratio[1] / (1 - split_ratio[0]))
    regular_df = regular_df.drop(regular_validation.index)
    regular_test = regular_df

    if(use_other):
        other_df = data[data["Label"] == "Other"]
        other_training = other_df.sample(frac=split_ratio[0])
        other_df = other_df.drop(other_training.index)
        other_validation = other_df.sample(frac=split_ratio[1] / (1 - split_ratio[0]))
        other_df = other_df.drop(other_validation.index)
        other_test = other_df

    covid_test = covid_df.sample(frac=covid_ratio)
    covid_validation = covid_df.drop(covid_test.index)


    if(use_other):
        training_data = pd.concat([bacterial_training, regular_training, other_training])
        validation_data = pd.concat(
            [bacterial_validation, regular_validation, other_validation, covid_validation]
        )
        test_data = pd.concat([bacterial_test, regular_test, other_test, covid_test])
    else:
        training_data = pd.concat([bacterial_training, regular_training])
        validation_data = pd.concat(
            [bacterial_validation, regular_validation, covid_validation]
        )
        test_data = pd.concat([bacterial_test, regular_test, covid_test])

    # Add a new column to the data to indicate the use of the image
    training_data["Use"] = "Train"
    validation_data["Use"] = "Val"
    test_data["Use"] = "Test"

    # Save the modified data
    modified_data_path = output_path
    pd.concat([training_data, validation_data, test_data]).to_csv(
        modified_data_path, index=False
    )

    training_ids = sorted(training_data["ID"].tolist())
    validation_ids = sorted(validation_data["ID"].tolist())
    test_ids = sorted(test_data["ID"].tolist())

    #Get absolute path of output path
    output_path = os.path.abspath(output_path)

    return {
        "file": output_path,
        "training": training_ids,
        "validation": validation_ids,
        "testing": test_ids,
        "ratios": {
            "training": split_ratio[0],
            "validation": split_ratio[1],
            "testing": split_ratio[2],
            "covid-testing": covid_ratio,
        },
        "counts": {
            "training": training_data.value_counts("Label").to_dict(),
            "validation": validation_data.value_counts("Label").to_dict(),
            "testing": test_data.value_counts("Label").to_dict(),
        }
    }
