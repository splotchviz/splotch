#!/usr/bin/env python

# Generate overlay images in PNG format with transparancy which can be
# used to label Splotch frames.  This script can be called as a
# standalone program, see below for details.  To label an entire
# directory of Splotch frames, use the driver script <splotchLabelFrames.sh>.
#
#                                         (Klaus Reuter, RZG, Sep 2011)


def splotchColormap(time=-1.0,               # for time>0, a time stamp is printed in the upper left corner
                    redshift=-1.0,           # for redshift>0, a redshift stamp is printed
                    valMin=0.1,              # minimum value for the log colorscale
                    valMax=1.e4,             # maximum value for the log colorscale
                    outfile="overlay.png",   # default file name of the overlay to be created
                    xinches=12,              # width of the image  | at 100 DPI, this corresponds to
                    yinches=8,               # height of the image | the dimensions 1200x800
                    myFontSize="large",
                    myFontColor="white",
                    putMinerva=False):       # place the MPG minerva logo in the top right corner

   # import necessary modules
   import numpy as np
   from matplotlib import pyplot
   import matplotlib as mpl
   from subprocess import call
   from math import pow

   # *** set font properties for annotations ***
   fprops=mpl.font_manager.FontProperties()
   fprops.set_size(myFontSize)
   #fprops.set_weight("bold")


   # *** set up the matplotlib colormap based on a Splotch colormap ***
   
   #$ cat OldSplotch.pal 
   #OldSplotch
   #0100
   #3
   #  0   0 255
   #128 255 128
   #255   0   0

   # See <http://matplotlib.sourceforge.net/api/colors_api.html>
   #  to understand what's going on ...
   # <OldSplotch.pal> corresponds to:
   OldSplotch = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
                 'blue':  ((0.0, 1.0, 1.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0))}
   colormap = mpl.colors.LinearSegmentedColormap('colormap', OldSplotch)
   # TODO implement a reader for Splotch palette files


   # *** set up the figure ***
   fig = pyplot.figure(figsize=(xinches,yinches))

   # *** set up the colorbar ***
   ax1  = fig.add_axes([0.90, 0.05, 0.02, 0.5])
   norm = mpl.colors.LogNorm(vmin=valMin, vmax=valMax)
   form = mpl.ticker.LogFormatterMathtext()
   cb1  = mpl.colorbar.ColorbarBase(ax1, cmap=colormap, norm=norm,
                                    format=form, orientation='vertical')
   # manipulate the style of the ticklabels, which requires a loop
   for tl in cb1.ax.get_yticklabels():
      tl.set_fontsize(myFontSize)
      tl.set_color(myFontColor)
   cb1.set_label('Temperature [K]', fontproperties=fprops, color=myFontColor)


   # *** set up the time/redshift variable ***
   if (time>=0.0):
      timeString="age of universe=%.3f" % (time, )
      timeString=timeString+" Gyr"
      pyplot.figtext(x=0.025, y=0.950, s=timeString, fontdict=None,
                     fontproperties=fprops, color=myFontColor)
   #
   if (redshift>0):
      timeString="redshift=%.3f" % (redshift, )
      pyplot.figtext(x=0.025, y=0.910, s=timeString, fontdict=None,
                     fontproperties=fprops, color=myFontColor)


   # Minerva needs an intermediate call of the ImageMagick tools
   if putMinerva:
      plotFile="./splotchColormapTmp.png"
   else:
      plotFile=outfile

   # *** finally, plot the image and write it to a png file ***
   pyplot.plot()
   F=pyplot.gcf()
   myDPI=100
   F.savefig(plotFile, transparent=True, dpi=myDPI)


   # *** put a logo (e.g. MPG Minerva) on top using ImageMagick convert ***
   if putMinerva:
      minervaFile="__INSERT_VALID_PATH__/minerva-white-96.png"
      xoffset=str(int( (xinches*myDPI)*0.895 ))
      yoffset=str(int( (yinches*myDPI)*0.005 ))
      #print (xoffset, yoffset)
      convertCommand="/usr/bin/env convert "+plotFile+" "+minervaFile+" -geometry +"+xoffset+"+"+yoffset+" -composite -format png "+outfile
      call(convertCommand, shell=True)  

   # *** END SplotchColormap() ***



#
# *** Allow this Python module to be run as a standalone script. ***
#
if __name__ == "__main__":
   import sys
   import getopt
   #
   try:
      opts, args = getopt.getopt(sys.argv[1:],
                        "t:r:c:d:o:",  # the "-" options, below are the "--" options
                        ["time=", "redshift=", "colormin=", "colormax=", "outfile="])
   except getopt.GetoptError, err:
      print str(err)
      sys.exit(2)
   #
   myOutFile  = "overlay.png"
   myTime     = -1.0
   myRedshift = -1.0
   myMinVal   = 1
   myMaxVal   = 100
   #
   for o, a in opts:
      # print (o,a)
      if   o in ("-t", "--time"):
         myTime     = float(a)
      elif o in ("-r", "--redshift"):
         myRedshift = float(a)
      elif o in ("-c", "--colormin"):
         myMinVal   = pow(10.0, float(a))
      elif o in ("-d", "--colormax"):
         myMaxVal   = pow(10.0, float(a))
      elif o in ("-o", "--outfile"):
         myOutFile  = a
      else:
          assert False, "unhandled option"
   #
   splotchColormap(outfile=myOutFile,
                   time=myTime,
                   redshift=myRedshift,
                   valMin=myMinVal,
                   valMax=myMaxVal)

# EOF

