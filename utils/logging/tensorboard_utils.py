import os

def delete_logs(logs_path):
    for the_file in os.listdir(logs_path):
        file_path = os.path.join(logs_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)