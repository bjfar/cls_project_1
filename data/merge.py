# Quick script to merge data files

def mergematch(oldfile,newfile,outfile,cols):
   print "Merging {0} into {1}, replacing old lines by matching first {2} columns".format(newfile,oldfile,cols)
   print "Output filename: {0}".format(outfile)
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
                     match=True
                     break
               if match==False:
                  f_merged.write(line_old)

# Do merges
mergematch("bchargino.dat","bCharginoAllExtra.dat","bchargino_merged.dat",3)
mergematch("mixed.dat","mixedextra.dat","mixed_merged.dat",3)

