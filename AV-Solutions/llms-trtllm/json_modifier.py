# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse

# Define a function to handle the argument and modify the JSON
def modify_json(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Modify the 'rotary_scaling' field
    data['rotary_scaling'] = None

    # Save the updated JSON back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Modified JSON has been saved to {file_path}")

# Set up argument parser
def main():
    parser = argparse.ArgumentParser(description="Modify a JSON config file.")
    parser.add_argument('config_file', type=str, help="Path to the JSON configuration file")
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the function with the provided config file path
    modify_json(args.config_file)

# Run the script if it's executed directly
if __name__ == "__main__":
    main()