#!/bin/bash

# Download matrix archives
wget -nv https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage4.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhda416.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/HB/mcfe.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Bai/olm1000.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Sandia/adder_dcop_32.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/HB/west2021.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity10.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Zitney/rdist2.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Simon/olafu.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Laplace.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhd4800a.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Simon/raefsky2.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Norris/lung2.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/PR02R.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal1.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt80.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/dc1.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_1_k101.tar.gz

wget -nv https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz

# Extract matrix files and delete archives
for filename in *.tar.gz
do
  tar -zxf "${filename}"
  rm "${filename}"
done

# Move files and delete folders
for dir in */
do
  f="${dir%*/}.mtx"
  mv "${dir}${f}" ./
  rm -r "${dir%*/}"
done