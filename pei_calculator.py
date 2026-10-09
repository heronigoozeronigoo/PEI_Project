# ==========================================================
# MATHEMATICAL PRIVACY EXPOSURE INDEX (PEI)
# Stage 1: PEI Calculator Prototype
# ==========================================================

# Sensitivity weights
# NOTE: These are temporary illustrative values.
weights = {
    "Name": 0.10,
    "Username": 0.10,
    "URL": 0.10,
    "Email": 0.15,
    "Phone": 0.15,
    "Account ID": 0.20,
    "QR Code": 0.20
}

# Temporary maximum raw exposure
# This will be finalized during validation.
R_MAX = 2.00


# ----------------------------------------------------------
# START PROGRAM
# ----------------------------------------------------------

print()
print("==================================================")
print("     MATHEMATICAL PRIVACY EXPOSURE INDEX")
print("              PEI CALCULATOR")
print("==================================================")
print()

print("This is Stage 1 of the research prototype.")
print("You will enter the detected sensitive information")
print("and the program will calculate the PEI.")
print()


# ----------------------------------------------------------
# NUMBER OF SENSITIVE ITEMS
# ----------------------------------------------------------

while True:

    try:
        number_of_items = int(
            input("How many sensitive items were detected? ")
        )

        if number_of_items >= 0:
            break

        print("Please enter 0 or a positive number.")

    except ValueError:
        print("Please enter a whole number.")


# Store total exposure
total_exposure = 0.0


# ----------------------------------------------------------
# PROCESS EACH SENSITIVE ITEM
# ----------------------------------------------------------

for i in range(number_of_items):

    print()
    print("------------------------------------------")
    print("Sensitive Item", i + 1)
    print("------------------------------------------")

    # Display choices
    print()
    print("Choose the type of sensitive information:")
    print("1 - Name")
    print("2 - Username")
    print("3 - URL")
    print("4 - Email")
    print("5 - Phone")
    print("6 - Account ID")
    print("7 - QR Code")

    # Information types
    types = {
        1: "Name",
        2: "Username",
        3: "URL",
        4: "Email",
        5: "Phone",
        6: "Account ID",
        7: "QR Code"
    }

    # Get valid type
    while True:

        try:
            choice = int(
                input("Enter the number (1-7): ")
            )

            if choice in types:
                break

            print("Please enter a number from 1 to 7.")

        except ValueError:
            print("Please enter a number from 1 to 7.")

    information_type = types[choice]

    # ------------------------------------------------------
    # AREA
    # ------------------------------------------------------

    while True:

        try:
            area = float(
                input("Relative area (0 to 1): ")
            )

            if 0 <= area <= 1:
                break

            print("Area must be between 0 and 1.")

        except ValueError:
            print("Please enter a decimal number.")

    # ------------------------------------------------------
    # VISIBILITY
    # ------------------------------------------------------

    while True:

        try:
            visibility = float(
                input("Visibility (0 to 1): ")
            )

            if 0 <= visibility <= 1:
                break

            print("Visibility must be between 0 and 1.")

        except ValueError:
            print("Please enter a decimal number.")

    # ------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------

    while True:

        try:
            confidence = float(
                input("Detection confidence (0 to 1): ")
            )

            if 0 <= confidence <= 1:
                break

            print("Confidence must be between 0 and 1.")

        except ValueError:
            print("Please enter a decimal number.")

    # ------------------------------------------------------
    # GET WEIGHT
    # ------------------------------------------------------

    weight = weights[information_type]


    # ------------------------------------------------------
    # CALCULATE ITEM SCORE
    #
    # s_i = w_i × A_i × V_i × C_i
    # ------------------------------------------------------

    item_score = (
        weight
        * area
        * visibility
        * confidence
    )


    # Add item score to total
    total_exposure += item_score


    # ------------------------------------------------------
    # DISPLAY ITEM RESULT
    # ------------------------------------------------------

    print()
    print("Item Information")
    print("------------------------------")
    print("Type:", information_type)
    print("Weight:", weight)
    print("Area:", area)
    print("Visibility:", visibility)
    print("Confidence:", confidence)

    print()
    print("Item Score Calculation:")
    print(
        weight,
        "×",
        area,
        "×",
        visibility,
        "×",
        confidence
    )

    print("Item Score =", round(item_score, 6))


# ----------------------------------------------------------
# CALCULATE PEI
#
# PEI = 100 × (R / R_MAX)
# ----------------------------------------------------------

pei = 100 * (total_exposure / R_MAX)


# Make sure PEI stays between 0 and 100
if pei < 0:
    pei = 0

if pei > 100:
    pei = 100


# ----------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------

if pei <= 33.33:

    classification = "LOW"

elif pei <= 66.67:

    classification = "MODERATE"

else:

    classification = "HIGH"


# ----------------------------------------------------------
# FINAL RESULTS
# ----------------------------------------------------------

print()
print()
print("==================================================")
print("                 FINAL RESULTS")
print("==================================================")

print()
print("Total Raw Exposure (R):")
print(round(total_exposure, 6))

print()
print("PEI Score:")
print(round(pei, 2), "/ 100")

print()
print("Privacy Exposure Level:")
print(classification)

print()
print("==================================================")

print()
print("The PEI calculation is:")
print("PEI = 100 × (R / R_MAX)")

print()
print("Thank you for using the PEI prototype.")
print()