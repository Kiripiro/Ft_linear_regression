from utils import load_thetas, estimated_price

blue_code = '\033[94m'
orange_code = '\033[93m'
red_code = '\033[91m'
end_code = '\033[0m'

def main():
    try:
        print(f"{blue_code}Please enter a mileage...{end_code}")
        file_path = 'raw_thetas.csv'
        theta0, theta1 = load_thetas(file_path)
        if theta0 == 0 and theta1 == 0:
            raise Exception("No thetas found, please train the model first")
        x = int(input("Enter mileage: "))
        price = estimated_price(theta0, theta1, x)
        if price < 0:
            raise Exception(f"{red_code}Mileage is too high, please enter a lower mileage.\nThe price would be negative.{end_code}")
        print(f"Estimated price for {orange_code}{x}km{end_code}: {blue_code}${round(price, 3)}{end_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()