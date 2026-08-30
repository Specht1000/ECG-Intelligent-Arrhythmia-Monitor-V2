"""Small dependency-free demonstration of 12-lead reconstruction."""

from ecg_v2 import reconstruct_12_leads


def main() -> None:
    independent_sample_uv = {
        "I": 120.0,
        "II": 310.0,
        "V1": -80.0,
        "V2": -35.0,
        "V3": 45.0,
        "V4": 160.0,
        "V5": 210.0,
        "V6": 185.0,
    }

    twelve_leads = reconstruct_12_leads(independent_sample_uv)
    print("One simultaneous sample (illustrative values in microvolts):")
    for name, value in twelve_leads.items():
        print("{:>3}: {:8.2f} uV".format(name, value))


if __name__ == "__main__":
    main()
