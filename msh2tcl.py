fname='r15ar5/refined.msh'
conversion=1e4
#conversion=1

msh=open(fname,"r")
tcl=open("msh2tcl.tcl","w")

L=msh.readlines()
ind=L.index("$Nodes\n")

tcl.write("color Display Background white\ndisplay depthcue off\ndisplay backgroundgradient off\ndisplay shadows on\ndisplay ambientocclusion on\ndisplay aoambient 0.6\ndisplay aodirect 0.6\n")



tcl.write("mol new atoms "+L[ind+1])
tcl.write("mol addrep top \nanimate dup top\n")

natoms=int(L[ind+1])
ind+=2

#elm-number elm-type number-of-tags < tag > … node-number-list
def readelement(L):
    num_tags=int(L[2])
    bondfile=L[2+num_tags+1:]
    if len(bondfile)>1: #and len(bondfile)<8:
        for i in range(len(bondfile)-1):
            tcl.write("topo addbond "+f'{int(bondfile[0])-1} {int(bondfile[1])-1}\n')

for i in L[ind:ind+natoms]:
    vertex=i.split()
    atom_index=str(int(vertex[0])-1)
    x=float(vertex[1])*conversion
    y=float(vertex[2])*conversion
    z=float(vertex[3])*conversion
    tcl.write("set sel [atomselect top \"index "+atom_index+"\"]\n")
    tcl.write("$sel set {x y z} {{"+f'{x} {y} {z}'+"}}\n")
ind=L.index("$Elements\n")+1

nE=int(L[ind])
for i in L[ind+1:ind+nE]:
    E=i.split()
    #readelement(E)
#tcl.write("mol representation QuickSurf 17 0.2 28\n")
tcl.write("mol addrep top\n")
tcl.write("set sel [atomselect top all]")
