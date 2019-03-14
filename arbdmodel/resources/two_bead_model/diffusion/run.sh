
for i in p b; do
    cp -f $i.hydropro.dat hydropro.dat
    ./hydropro10-lnx.exe 
done
