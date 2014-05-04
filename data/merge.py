# Quick script to merge data files

def mergematch(oldfile,newfile,outfile,cols):
   print "Merging {0} into {1}, replacing old lines by matching first {2} columns".format(newfile,oldfile,cols)
   print "Output filename: {0}".format(outfile)
   donelines = []  #when we write out a line of the newfile, record its line number here
   with open(outfile,"w") as f_merged:
      with open(oldfile,"r") as f_old:
         with open(newfile,"r") as f_new:
            for i,line_old in enumerate(f_old):
               match=False
               oldmasses = line_old.split()[:cols]
               f_new.seek(0) # Go back to start of 'new' file
               for j,line_new in enumerate(f_new):
                  newmasses = line_new.split()[:cols]
                  if oldmasses==newmasses:
                     print "    Replacing line {0} of old data with line {1} of new data".format(i,j)
                     f_merged.write(line_new)
                     donelines+=[j]
                     match=True
               if match==False:
                  f_merged.write(line_old)
            # Old file written out with replacements from new file
            # Now need to add any extra lines from newfile to the merged file
            with open(newfile,"r") as f_new:
               f_new.seek(0) # Go back to start of 'new' file
               print "    Adding new points from new file to the merged file..."
               for k,line_new in enumerate(f_new):
                  if k not in donelines:
                     print "    Adding line {0} of new data to merged file".format(k)
                     f_merged.write(line_new)
           
# Do merges
mergematch("bchargino.dat","bCharginoAllExtra.dat","bchargino_merged_tmp.dat",3)   #first set of extra points
mergematch("bchargino_merged_tmp.dat","bcharginoExtra2.dat","bchargino_merged_tmp2.dat",3) #second set of extra points
mergematch("bchargino_merged_tmp2.dat","bcharginoFinal.dat","bchargino_merged.dat",3) #third set of extra points

mergematch("mixed.dat","mixedextra.dat","mixed_merged_tmp.dat",3)
mergematch("mixed_merged_tmp.dat","mixedfinal.dat","mixed_merged_tmp2.dat",3)
mergematch("mixed_merged_tmp2.dat","mixedfinal_2.dat","mixed_merged.dat",3)
