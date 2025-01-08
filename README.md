# Sentiment-Analysis


# How to Run:
## CPU
To run the CPU file, navigate to the root directory (i.e. cd downloads/cpu/hellocpp/src/ or replace downloads with wherever you place the folder), compile with g++ -std=c++17 *.cpp, then run ./a.out or a.exe depending on your OS. "Program started." will print to the terminal and it will take about 3-5 minutes to finish the entire inference. There may be issues with running this code on Windows, so MacOS is preferred. If this doesn't work on either of those systems, the FPGA code below runs on Linux along with MacOS.

## FPGA
For FPGA, delete all .o files in the obj folder, delete ForwardPass.out/exe, and then navigate to the root folder (Inside) and simply type "make". then just type ./ForwardPass. You will have to change the file path of the test.csv file to match it's location on your machine.

## GPU
Link for GPU code, also added as .ipynb file to github:
<br>
https://colab.research.google.com/drive/1YuLHUc2aOO4L9Tx0SThb9z69SOQaj2vM?usp=sharing
