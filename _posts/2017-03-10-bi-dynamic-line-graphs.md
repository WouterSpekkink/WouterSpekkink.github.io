---
layout: post
title:  "Bi-dynamic Line Graphs"
date:   2017-03-10 14:25:10 +0000
categories: Software 
tags: Non-Technical
---

# Introduction
This program is based on an idea first proposed in an article by Chiara Broccatelli, Martin Everett, and Johan Koskinen, entitled *Temporal dynamics in covert networks* and published in 2016 in the journal *Methodological Innovations* (vol 9, pp. 1-14). The program was written by Wouter Spekkink. All mistakes that might exist in the program are his. 


# Downloads
You can download the program using the following links:

   * [Download for Linux](https://github.com/WouterSpekkink/BDLG_Matrix_Converter/releases/download/BDLG_Matrix_Converter_Linux_V1.0.1/BDLG_Matrix_Converter_Linux_v1.0.1.tar.gz)
  
   * [Download for Windows](https://github.com/WouterSpekkink/BDLG_Matrix_Converter/releases/download/BDLG_Matrix_Converter_Windows_V1.0.1/BDLG_Matrix_Converter_Windows_v1.0.1.zip)

# Bi-dynamic line graphs
In their article, Brocatelli et al. (2016) introduce bi-dynamic line-graph (BDLG) projections of social processes to study dynamics in social networks. The BDLG can be understood as a special projection of a two-mode network of actors and events. In matrix form, a two-mode network is usually presented as an incidence matrix with actors in rows, and events in columns, like so (example based on article mentioned above):

|   | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A | 1 | 1 | 1 | 0 | 0 |
| B | 0 | 1 | 1 | 1 | 0 |
| C | 0 | 1 | 0 | 1 | 1 |
| D | 0 | 0 | 0 | 0 | 1 |

In Social Network Analysis, it is common to convert two-mode networks to one-mode networks that indicate which actors participated in the same events (and how many times they participated in events together). This is done by multiplying the incidence matrix to a weighted adjacency matrix, which can be achieved by converting the incidence matrix with a transposed version of itself. If we would do that for the incidence matrix above, we would get the following adjacency matrix:

|   | A | B | C | D |
|---|---|---|---|---|
| A | 3 | 2 | 1 | 0 |
| B | 2 | 3 | 2 | 0 |
| C | 1 | 2 | 3 | 1 |
| D | 0 | 0 | 1 | 1 |

Although this gives insight in the ties that emerge from the events in which actors participate, the drawback is that all temporal information is lost; only a static network remains. This can be solved partially by making separate adjacency matrices for different segments of the incidence matrix, but the choice for the size of the segments is always arbitrary, and information about changes within each segment is still lost (see Brocatelli et al. 2016 for a discussion on this).

Brocatelli et al. (2016) propose BDLGs as an alternative projection of two-mode networks that have a temporal dimension. In a BDLG the edges between actors and events of the original two-mode graph are 'converted' to nodes. For example, the edge between actor *A* and event *1* in the original two-mode graph becomes the node *A1* in the BDLG. In a BDLG, two types of relationships can exist between these nodes. Mutual (or reciprocal) edges only exist between nodes that indicate actors that are involved in the same event (e.g., *A1*<->*B1*). Edges with only one direction indicate the continued presence of actors throughout events (e.g., *A1*->*A2*->*A4*). To prevent the inclusion of redundant information in the BDLG, Brocatelli et al. (2016) also introduce the rule that transitive paths are always reduced to a single path. For example, if actor *A* was involved in events *1*, *2*, and *4*, it is only necessary to create the edges *A1*->*A2* and *A2*->*A4*. The edge *A1*->*A4* introduces redundant information about the continued involvement of actor *A*, and can therefore be excluded. This rule leads to more elegant presentations that allow researchers to trace actors' involvement in the studied process. The matrix form of the BDLG thus becomes as follows (rows with only *0* values, as well as their corresponding columns, have been removed from the matrix):

||A1|A2|A3|B2|B3|B4|C2|C4|C5|D5|
|---|---|---|---|---|---|---|---|---|---|---|
|A1|0|1|0|0|0|0|0|0|0|0|
|A2|0|0|1|1|0|0|1|0|0|0|
|A3|0|0|0|0|1|0|0|0|0|0|
|B2|0|1|0|0|1|0|1|0|0|0|
|B3|0|0|1|0|0|1|0|0|0|0|
|B4|0|0|0|0|0|0|0|1|0|0|
|C2|0|1|0|0|0|0|0|1|0|0|
|C4|0|0|0|0|0|1|0|0|1|0|
|C5|0|0|0|0|0|0|0|0|0|1|
|D5|0|0|0|0|0|0|0|0|1|0|

# The program
Based on the ideas outlined above, Wouter Spekkink wrote a simple tool that converts incidence matrices into BDLG matrices. The program expects the user to import a csv-file with the incidence matrix of the two-mode network. The program assumes that the columns of the incidence matrix are ordered in time from left to right. The order of the actors is unimportant. 

The tool can export the matrix itself to a csv-file. In addition, the tool can export a node list and edge list, where the edge list contains the same information as the matrix, whereas the nodes list also includes some variables that indicate, for each node, the event that is part of the node (*Order_Original*), the actor that is part of the node (*Actor*), and the order in which the events occurred (*Order_Closed*). 

# Running the program
The Windows version of the program can be run by simply opening the application. The Linux version can be run by using the accompanying shell script. It might be necessary to change permissions for the shell script first, by typing the following in the console:

	> sudo chmod+x ./bdlg-matrix.sh

# Using the program with Gephi
A quick way to visualise the results is to use the tool to export the node list and edge list, and import these into [Gephi](http://www.gephi.org). Importing of the edge list and node list should be done from Gephi's data laboratory. The user should make sure that the order variables of the node list are imported as numeric variables (doubles are the safest option, but integers should also work in most situations). To make useful plots of the results, the user should make sure to also install the Event Graph Layout (this can be done from within the Plugins menu of Gephi). The user should then select one of the order variables as the order variable for the Event Graph Layout plugin. The *Actor* variable can be used to give nodes colours based on the actors associated with them. This allows one to more easily spot lineages in the graph. 

# Dependencies
The program was written in C++, and with the help of Qt5 libraries. Thus, Qt5 must be installed on your system if you wish to build the program yourself. 


