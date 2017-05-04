

insert_clause = INSERT INTO patient_data (gender, firstname, lastname, street, city, region, zipcode, country, phone , latitude, longtitude, sin_lat, cos_lat, rad_lon ) VALUES


def add_data():
    records_to_add = raw_input('Give me records to add!')
    return insert_clause + records_to_add



if __name__ == '__main__':
    add_data()
