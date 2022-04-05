# LHFiD

This repository contains the source codes for a many-objective evolutionary algorithm (MaOEA), namely LHFiD, its variant LHFiD-i, and some competitor MaOEAs. A manuscript based on LHFiD (and LHFiD-i) has been submitted to IEEE Transactions on Evolutionary Computation, and is under review.

# Contents

The MaOEAs included in this repository, along with their underlying platforms, are:
a) LHFiD and LHFiD-i (pymoo)  
b) NSGA-III and NSGA-III-i (pymoo)  
c) θ-DEA and θ-DEA-i (PlatEMO)  
d) RPD-NSGA-II and RPD-NSGA-II-i (PlatEMO)  
e) MOEA/D-LWS and MOEA/D-LWS-i (pymoo)  
f) multiGPO (PlatEMO)  
g) MOEA/AD (jmetal)  
The relevant description, references and parameter settings, for the above MaOEAs can be found in our submitted manuscript. Notably, pymoo is a python-based platform for developing MaOEAs. Similarly, PlatEMO is MATLAB-based and jmetal is java-based. The difference in the platforms of above MaOEAs is due to the availability of their respective source codes.

Further, the test problems used in the manuscript include: (a) DTLZ, (b) WFG, (c) MaF, (d) minus-DTLZ, and (e) minus-WFG. None of the above platforms (pymoo, PlatEMO or jmetal) covers all these test problems, owing to which, the source codes for the missing test problems have also been included in this repository. 

# Requirements

For running the MaOEAs built on pymoo, make sure you have the following installed:
a) Python 3.6 or above: https://www.python.org/
b) pymoo package, as available on PyPi and can be installed by: pip install pymoo==0.4.2.2rc2
c) hvwfg package, as available on PyPi and can be installed using: pip install hvwfg

For running the MaOEAs built on PlatEMO, make sure you have the following installed:
a) MATLAB R2019a or above: https://www.mathworks.com/products/get-matlab.html?s_tid=gn_getml
b) PlatEMO 2.7 (source code included in this repository)

For running the MaOEAs built on jmetal, make sure you have the following installed:
a) Java development kit (JDK) 8 or above: https://www.oracle.com/java/technologies/downloads/
b) Netbeans IDE 8.2 or above: https://netbeans-ide.informer.com/8.2/

# Installation from this Repository

For MaOEAs built on pymoo:
a) download the respective file from “MaOEAs/pymoo” folder,
b) locate the “pymoo/algorithms” folder in your python installation directory, and
c) copy the downloaded file(s) in that folder.

For MaOEAs built on PlatEMO:
a) download the respective folders from “MaOEAs/PlatEMO” directory,
b) locate the “algorithms” folder in your PlatEMO installation directory, and
c) copy the downloaded file(s) in that folder.

For MaOEAs built on jmetal:
a) download the respective file from “MaOEAs/jmetal” directory,
b) extract the project from the file any “.rar” extractor, and
c) import the extracted folder as a project in NetBeans.

# Instructions on how to run

For MaOEAs built on pymoo, a “.ipynb” notebook file has been provided in the “Run” folder. In that:
a) the problem name, number of objectives, and number of generations (except for LHFiD and LHFiD-i) need to be specified at the top, and
b) all cells are to be executed.
Notably, the “Problems/pymoo/maf.py” file should be copied to the current working directory.

The MaOEAs built on PlatEMO can be executed through direct commands, as documented in the “manual” provided in the PlatEMO source file.

Only one MaOEA (MOEA/AD) is built on jmetal, for which the executable file has been provided in the “Run” folder.

If you have any difficulties running these codes, please write to dhish.saxena@me.iitr.ac.in or smittal1@me.iitr.ac.in
