import json

notebook = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, glob\n",
    "import pandas as pd\n",
    "\n",
    "def find_data_dirs():\n",
    "    kaggle_input = '/kaggle/input'\n",
    "    if os.path.exists(kaggle_input):\n",
    "        for root, dirs, files in os.walk(kaggle_input):\n",
    "            if 'train' in dirs and 'test' in dirs:\n",
    "                return os.path.join(root, 'train'), os.path.join(root, 'test')\n",
    "    return None, None\n",
    "\n",
    "_, test_dir = find_data_dirs()\n",
    "if test_dir:\n",
    "    test_files = sorted(glob.glob(os.path.join(test_dir, '*_horizontal_well.csv')))\n",
    "    for f in test_files:\n",
    "        df = pd.read_csv(f)\n",
    "        print(f'=== Test Well {os.path.basename(f)} ===')\n",
    "        print('Columns:', list(df.columns))\n",
    "        print('Shape:', df.shape)\n",
    "        known = df[df.TVT_input.notna()]\n",
    "        print(f'Known rows: {len(known)}')\n",
    "        print('First 3 known rows:')\n",
    "        print(known.head(3)[['MD', 'Z', 'TVT_input']].to_string())\n",
    "        print('Last 3 known rows:')\n",
    "        print(known.tail(3)[['MD', 'Z', 'TVT_input']].to_string())\n",
    "        print()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("predict_tvt.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("[+] Notebook created successfully at predict_tvt.ipynb")
