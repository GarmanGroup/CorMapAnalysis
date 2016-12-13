"""CorMap class implementing analyis of results from the ATSAS suite program
DATCMP.
"""
import glob
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns


class ScatterAnalysis(object):
    """
    Scatter curve similarity analysis class.

    CorMap class implementing analysis of data output from a run of the
    ATSAS suite program DATCMP.
    """

    # ----------------------------------------------------------------------- #
    #                         CLASS VARIABLES                                 #
    # ----------------------------------------------------------------------- #
    PLOT_LABEL = {'family': 'serif',
                  'weight': 'normal',
                  'size': 16}
    mpl.rc('font', family='serif', weight='normal', size=12)

    PLOT_NUM = 0

    # ----------------------------------------------------------------------- #
    #                         CONSTRUCTOR METHOD                              #
    # ----------------------------------------------------------------------- #
    def __init__(self, scat_curve_location, x_axis_vec=[], x_metric="",
                 x_units=""):
        """
        Create a ScatterAnalysis object.

        Constructor method to create ScatterAnalysis object which exposes a
        simple API for analysing the DATCMP data.

        Parameters
        ----------
        scattering_curve_location : str
            Location of the scattering curves for the dataset.
        x_axis_vec : numpy array, optional (default=[])
            Vector of values to use on the x-axis of plots. This can be things
            like dose values for example. If not specified then frame number is
            used as a default.
        x_metric : str, optional (default="")
            String specifying the type of metric used for the x-axis. An
            example would be "Dose". If not specified then the default metric
            used on the x-axis is "Frame Number"
        x_units : str, optional (default="")
            String specifying the units used for the x-axis metric. For example
            if the x-axis metric was the 'dose', then x_units may be specified
            as "kGy".

        Returns
        -------
        ScatterAnalysis
            The ScatterAnalysis object

        Examples
        --------
        To create a ScatterAnalysis object that will plot only against frame
        numbers then the only input required is the location of the set of
        files:

        >>>  scat_obj = ScatterAnalysis("saxs_files.00*.dat")

        If you want to plot an x-axis that is different from the frame number,
        e.g. the dose, then we would specify something like:

        >>>  scat_obj = ScatterAnalysis("saxs_files.00*.dat", np.array([10, 20, 30, 40]), "Dose", "kGy")

        Note
        ----
        The x_axis_vec array needs to be the same length as the total number of
        frames in the dataset otherwise the frame number will be used on the
        x-axis by default

        """
        # Go through files and extract the frame data.
        file_list = glob.glob(scat_curve_location)
        num_frames = len(file_list)
        self.q = np.loadtxt(file_list[0])[:, 0]
        self.I = np.zeros([len(self.q), num_frames])
        for i, file in enumerate(file_list):
            frame_data = np.loadtxt(file)
            self.I[:, i] = frame_data[:, 1]

        # Run DATCMP to get pairwise comparison information.
        self.datcmp_data = self.get_datcmp_info(scat_curve_location)

        # Organise the x-axis used for the plots. Default will be the frame
        # number.
        if not isinstance(x_axis_vec, list):
            if len(x_axis_vec) == num_frames:
                self.x_axis = x_axis_vec
            else:
                print "x_axis_vec is not the same length as the number of"
                print "frames. Using frame numbers instead."
                self.x_axis = np.linspace(1, num_frames, num_frames)
        else:
            self.x_axis = np.linspace(1, num_frames, num_frames)

        if x_metric and x_units:
            self.x_metric = x_metric
            self.x_units = x_units
        else:
            self.x_metric = "Frame number"
            self.x_units = ""

    # ----------------------------------------------------------------------- #
    #                         INSTANCE METHODS                                #
    # ----------------------------------------------------------------------- #

    def get_datcmp_info(self, scattering_curve_files):
        """
        Extract the data produced by DATCMP.

        This method is used by the constructor method of the ScatterAnalysis
        class. It runs the DATCMP program on the dataset and returns a
        dictionary containing the main results from the DATCMP run.

        Parameters
        ----------
        scattering_curve_files : str
            Location of the scattering curves for the dataset.

        Returns
        -------
        dict(str, numpy.array)
            Dictionary containing the results of the DATCMP run. The dictionary
            key is a string (with no spaces) denoting the pair of frames that
            were compared e.g. "1,2" would be frames 1 and 2. The dictionary
            value is an array of DATCMP results for the corresponding pairwise
            comparison.

        Examples
        --------
        >>>  datcmp_data = scat_obj.get_datcmp_info("saxs_files.00*.dat")
        """
        cmd = "datcmp {}".format(scattering_curve_files)
        log = run_system_command(cmd)
        # define a dictionary to store the data produced from DATCMP - this
        # value will be overwritten.
        data_dict = {"1,2": 0}
        for line in iter(log.splitlines()):
            match_obj = re.match(r'\s* \d{1,} vs', line)
            if match_obj:
                data = line.split()
                if "*" in data[5]:
                    data[5] = data[5][:-1]
                data_dict["{},{}".format(data[0], data[2])] = [int(float(data[3])),
                                                               float(data[4]),
                                                               float(data[5])]
        return data_dict

    def find_diff_frames(self, frame=1, P_threshold=0.01, P_type="adjP"):
        """
        List all statistically dissimilar frames.

        This method finds all statistically dissimilar frames to any given
        frame using the CorMap test as outlined in Daniel Franke, Cy M Jeffries
        & Dmitri I Svergun (2015). The user can set the significance threshold
        as well as whether to use the Bonferroni corrected P values or not.
        (we recommend that you should use the Bonferroni corrected P values).

        Parameters
        ----------
        frame : int, optional (default=1)
            The frame that every other frame in the dataset is compared with.
            If not specified then all frames are compared to the first frame.
        P_threshold : float, optional (default=0.01)
            The significance threshold of the test. If it's not given then the
            default value is 1%.
        P_type : str, optional (default="adjP")
            String denoting whether to use the Bonferroni corrected P value
            (input string="adjP") or the ordinary P value (input string="P").
            Default is to use the Bonferroni corrected P value.

        Returns
        -------
        List
            A list of integers corresponding to all of the dissimilar frames

        Examples
        --------
        Find all frames that are dissimilar to frame 10

        >>>  diff_frames = scat_obj.find_diff_frames(frame=10)
        """
        if P_type == "adjP":
            p_col = 2
        elif P_type == "P":
            p_col = 1
        else:
            print "********************** ERROR ***************************"
            print "P_type '{}' Is not recognised".format(P_type)
            print "P_type can only take the values 'adjP' (default) or 'P'."
        if frame <= self.I.shape[1]:
            diff_frames = []
            for i in xrange(0, self.I.shape[1]):
                if i+1 < frame:
                    key = "{},{}".format(i+1, frame)
                elif i+1 > frame:
                    key = "{},{}".format(frame, i+1)
                else:
                    continue
                significance_val = self.datcmp_data[key][p_col]
                if significance_val < P_threshold:
                    diff_frames.append(i+1)
            return diff_frames
        else:
            print "********************** ERROR ***************************"
            print "FRAME '{}' DOES NOT EXIST".format(frame)
            print "Use different frame numbers between 1 and {}".format(self.I.shape[1])

    def find_first_n_diff_frames(self, n=1, frame=1, P_threshold=0.01,
                                 P_type="adjP"):
        """
        Find the first of n consecutive dissimilar frames.

        Return the first frame, F, where there are n-1 consecutive frames
        after F that are also statistically dissimilar from the chosen frame.

        Parameters
        ----------
        n : int, optional (default=1)
            The number of consecutive dissimilar frames to be considered
            significant.
        frame : int, optional (default=1)
            The frame that every other frame in the dataset is compared with.
            If not specified then all frames are compared to the first frame.
        P_threshold : float, optional (default=0.01)
            The significance threshold of the test. If it's not given then the
            default value is 1%.
        P_type : str, optional (default="adjP")
            string denoting whether to use the Bonferroni corrected P value
            (input string="adjP") or the ordinary P value (input string="P").
            Default is to use the Bonferroni corrected P value.

        Returns
        -------
        int
            Frame number of the first of n consecutive dissimilar frames.

        Examples
        --------
        Get the frame number that is the first of 3 consecutive dissimilar
        frames to frame 10.

        >>>  first_diff_frames = scat_obj.find_first_n_diff_frames(n=3, frame=10)
        """
        # Get list frames that are different
        list_of_diff_frames = self.find_diff_frames(frame, P_threshold, P_type)
        if n == 1:
            # If only looking for one frame then return the first value in
            # list of different frames.
            return list_of_diff_frames[0]
        elif n > 1:
            # If we're looking for more than one consecutive frame then we need
            # to keep track of the number of consecutive frames that we've
            # iterated through.
            consec_count = 0
            max_consec_count = 0
            fr_max_count = 0
            for i, curr_fr in enumerate(list_of_diff_frames):
                if i == 0:
                    prev_fr = curr_fr
                    consec_count = 1
                else:
                    if curr_fr == prev_fr + 1:
                        consec_count += 1
                    else:
                        consec_count = 1
                prev_fr = curr_fr
                if consec_count == n:
                    return curr_fr - n + 1
                if consec_count > max_consec_count:
                    max_consec_count = consec_count
                    fr_max_count = curr_fr - max_consec_count + 1
            print "************************ WARNING **************************"
            print "{} consecutive frames not reached!".format(n)
            print "The max number of consecutive frames was {}".format(max_consec_count)
            print "The initial frame for that run was frame {}.".format(fr_max_count)
        else:
            print "********************** ERROR ***************************"
            print "n MUST BE A POSITVE INTEGER VALUE"
            print "User chose n = {}.".format(n)
            print "Please choose a positve integer value for n."

    def similar_frames(self, frame=1, P_threshold=0.01, P_type="adjP"):
        """
        List all statistically similar frames.

        This method finds all statistically similar frames to any given
        frame using the CorMap test as outlined in Daniel Franke, Cy M Jeffries
        & Dmitri I Svergun (2015). The user can set the significance threshold
        as well as whether to use the Bonferroni corrected P values or not.
        (we recommend that you should use the Bonferroni corrected P values).

        Parameters
        ----------
        frame : int, optional (default=1)
            The frame that every other frame in the dataset is compared with.
            If not specified then all frames are compared to the first frame.
        P_threshold : float, optional (default=0.01)
            The significance threshold of the test. If it's not given then the
            default value is 1%.
        P_type : str, optional (default="adjP")
            string denoting whether to use the Bonferroni corrected P value
            (input string="adjP") or the ordinary P value (input string="P").
            Default is to use the Bonferroni corrected P value.

        Returns
        -------
        List
            A list of integers corresponding to all of the similar frames

        Examples
        --------
        Find all frames that are similar to frame 10

        >>>  similar_frames = scat_obj.similar_frames(frame=10)
        """
        list_of_diff_frames = self.find_diff_frames(frame, P_threshold, P_type)
        return [i+1 for i in xrange(0, self.I.shape[1]) if i+1 not in list_of_diff_frames]

    def get_pw_data(self, frame1, frame2, datcmp_data_type="adj P(>C)"):
        """
        Get the CorMap results for a given pair of frames.

        Return C, P(>C) or Bonferroni adjusted P(>C) value from the DATCMP
        output.

        Parameters
        ----------
        frame1 : int
            number of the 1st frame used for the pairwise analyis
        frame2 : int
            number of the 2nd frame used for the pairwise analyis
        datcmp_data_type: str, optional (default="adj P(>C)")
            String specifying the pairwise result to be returned.
            The input options are:
            1) 'C' - This will return the C value i.e. the max observed patch
            of continuous runs of -1 or 1
            2) 'P(>C)' - This will return the P value of observing a patch of
            continuous runs of -1 or 1 bigger than the corresponding C value.
            3) 'adj P(>C)' - This will return the Bonferroni adjusted P value
            of observing a patch of continuous runs of -1 or 1 bigger than the
            corresponding C value.

        Returns
        -------
        int or float
            A value specifying the largest number of consecutive positive or
            negative values in a row, C, or the probability value (either
            standard P or Bonferroni corrected) the we observed more than C
            positive or negative values in a row.

        Examples
        --------
        Get the number of the longest run of positive or negative values in a
        row between frames 1 and 2

        >>>  C = scat_obj.get_pw_data(1, 2, datcmp_data_type='C')

        Get the Bonferroni corrected probability that we would observed a run
        of positive or negative values greater than C

        >>>  C = scat_obj.get_pw_data(1, 2, datcmp_data_type='adj P(>C)')
        """
        if datcmp_data_type == "C" or datcmp_data_type == 0:
            dat_type = 0
        elif datcmp_data_type == "P(>C)":
            dat_type = 1
        elif datcmp_data_type == "adj P(>C)":
            dat_type = 2
        else:
            print "********************** ERROR ***************************"
            print "INVALID DATCMP DATA TYPE CHOSEN: '{}' DOES NOT EXIST".format(datcmp_data_type)
            print "Please choose either 'C', 'P(>C)' or 'adj P(>C)'."

        if frame1 < frame2:
            datcmp_key = "{},{}".format(frame1, frame2)
        elif frame2 < frame1:
            datcmp_key = "{},{}".format(frame2, frame1)

        if datcmp_key in self.datcmp_data:
            return self.datcmp_data[datcmp_key][dat_type]
        else:
            print "********************** ERROR ***************************"
            print "KEY '{}' DOES NOT EXIST".format(datcmp_key)
            print "Use different frame numbers between 1 and {}".format(self.I.shape[1])

    def calc_cormap(self):
        """
        Calculate correlation map.

        Method to calculate the full correlation map for entire dataset.

        Parameters
        ----------
        N/A

        Returns
        -------
        numpy 2D array
            2D matrix of correlations for all SAXS frames

        Examples
        --------
        >>>  correlation_matrix = scat_obj.calc_cormap()
        """
        return np.corrcoef(self.I)

    def calc_pwcormap(self, frame1, frame2):
        """
        Calculate pairwise correlation map between two frames.

        Calculation of the pairwise correlation map between two chosen frames.

        Parameters
        ----------
        frame1 : int
            Number of the 1st frame used for the pairwise analyis
        frame2 : int
            Number of the 2nd frame used for the pairwise analyis

        Returns
        -------
        2D Numpy array
            Array with the correlation between two frames.

        Examples
        --------
        Get the correlation map between frames 1 and 10.

        >>>  pairwise_corr_map = scat_obj.calc_pwcormap(1, 10)
        """
        pw_I = np.column_stack([self.I[:, frame1-1], self.I[:, frame2-1]])
        return np.corrcoef(pw_I)

    def get_pw_data_array(self, frame=0, delete_zero_row=True):
        """
        Get C, P(>C) and Bonferroni corrected P(>C) values for a given frame.

        Return an array of all C, P(>C) and Bonferroni adjusted P(>C) values
        from the DATCMP output for the requested frame. E.g. if you choose
        frame 1 then this means that the method with return an array of C,
        P(>C) and Bonferroni adjusted P(>C) values calculated between frame 1
        and all other frames.

        Parameters
        ----------
        frame : int, optional (default=0)
            The frame that every other frame in the dataset is compared with.
            If not specified then ALL data from the DATCMP results are
            returned.
        delete_zero_row : bool, optional (default=True)
            Choose whether to include a row where the chosen frame would've
            been compared with itself - although this row is filled with zero.
            This makes sure that you'll return an array whose number of rows
            is the same as the number of frames. By default this is set to
            false.

        Returns
        -------
        2D Numpy array
            Array of all C, P(>C) and Bonferroni adjusted P(>C) values from the
            DATCMP results for the requested frame.

        Examples
        --------
        Get all DATCMP data for when all other frames are compared with frame
        10

        >>>  datcmp_data = scat_obj.get_pw_data_array(frame=10)
        """
        if frame == 0:
            pw_data = np.zeros([len(self.datcmp_data), 3])
            for i, values in enumerate(self.datcmp_data.itervalues()):
                pw_data[i, :] = np.asarray(values)
        elif 1 <= frame <= self.I.shape[1]:
            pw_data = np.zeros([self.I.shape[1], 3])
            for i in xrange(0, self.I.shape[1]):
                if i+1 < frame:
                    key = "{},{}".format(i+1, frame)
                elif i+1 > frame:
                    key = "{},{}".format(frame, i+1)
                else:
                    continue
                pw_data[i, :] = np.asarray(self.datcmp_data[key])
        else:
            print "********************** ERROR ***************************"
            print "FRAME '{}' DOES NOT EXIST".format(frame)
            print "Use a frame number between 1 and {}".format(self.I.shape[1])

        if delete_zero_row and frame > 0:
            return np.delete(pw_data, (frame-1), axis=0)
        else:
            return pw_data


# ----------------------------------------------------------------------- #
#                        PLOT THE CORRELATION MAP                         #
# ----------------------------------------------------------------------- #
    def plot_cormap(self, colour_scheme="gray", display=True, save=False,
                    filename="", directory=""):
        """
        Plot the full correlation map.

        Plot the correlation map of all frames in the dataset.

        Parameters
        ----------
        colour_scheme : str, optional (default="gray")
            The colour scheme used to plot the correlation map. The list of
            schemes can be found on the relevant matplotlib webpage:
            http://matplotlib.org/examples/color/colormaps_reference.html
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.

        Examples
        --------
        Plot the full correlation map

        >>>  scat_obj.plot_cormap()

        Save the correlation map in the current working directory without
        displaying it

        >>>  scat_obj.plot_cormap(display=False, save=True, filename="MyPlot.png")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        min_q = min(self.q)
        max_q = max(self.q)
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          PLOT CORMAP                            #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        plt.figure(self.PLOT_NUM)
        plt.gca().xaxis.grid(False)
        plt.gca().yaxis.grid(False)
        cormap = plt.imshow(self.calc_cormap(), cmap=colour_scheme,
                            extent=[min_q, max_q, min_q, max_q])
        plt.xlabel(r'Scattering Vector, q (nm$^{-1}$)',
                   fontdict=self.PLOT_LABEL)
        plt.ylabel(r'Scattering Vector, q (nm$^{-1}$)',
                   fontdict=self.PLOT_LABEL)
        plt.colorbar(cormap)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

# ----------------------------------------------------------------------- #
#                    PLOT THE PAIRWISE CORRELATION MAP                    #
# ----------------------------------------------------------------------- #
    def plot_pwcormap(self, fr1, fr2, colour_scheme="gray", display=True,
                      save=False, filename="", directory=""):
        """
        Plot the pairwise correlation map between two chosen frames.

        Plot the pairwise correlation map between two frames fr1 and fr2.

        Parameters
        ----------
        fr1 : int
            Number of the 1st frame used for the pairwise analyis
        fr2 : int
            Number of the 2nd frame used for the pairwise analyis
        colour_scheme : str, optional (default="gray")
            The colour scheme used to plot the correlation map. The list of
            schemes can be found on the relevant matplotlib webpage:
            http://matplotlib.org/examples/color/colormaps_reference.html
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.

        Examples
        --------
        Plot the pairwise correlation map between frame 1 and frame 2

        >>>  scat_obj.plot_pwcormap(1, 2)

        Save the pairwise correlation map in the current working directory
        without displaying it

        >>>  scat_obj.plot_pwcormap(1, 2, display=False, save=True, filename="MyPlot.png")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        min_q = min(self.q)
        max_q = max(self.q)
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                     PLOT PAIRWISE CORMAP                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        fig = plt.figure(self.PLOT_NUM)
        plt.gca().xaxis.grid(False)
        plt.gca().yaxis.grid(False)
        cormap = plt.imshow(self.calc_pwcormap(frame1=fr1, frame2=fr2),
                            cmap=colour_scheme,
                            extent=[min_q, max_q, min_q, max_q])
        plt.xlabel(r'Scattering Vector, q (nm$^{-1}$)',
                   fontdict=self.PLOT_LABEL)
        plt.ylabel(r'Scattering Vector, q (nm$^{-1}$)',
                   fontdict=self.PLOT_LABEL)
        plt.colorbar(cormap)
        adjP = self.get_pw_data(fr1, fr2, "adj P(>C)")
        C = self.get_pw_data(fr1, fr2, "C")
        if self.x_units:
            change_in_x = abs(self.x_axis[fr1-1] - self.x_axis[fr2-1])
            plt.title(r'PW CorMap: frame {} vs {}.C = {}, adj P(>C) = {}, $\Delta${} = {:.2f} {}'.format(fr1, fr2, C, adjP, self.x_metric, change_in_x, self.x_units), ha='center')
        else:
            plt.title(r'Pairwise CorMap: frame {} vs {}. C = {}, adj P(>C) = {}'.format(fr1, fr2, C, adjP), ha='center')

        fig.canvas.mpl_connect('draw_event', on_draw)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

    # ----------------------------------------------------------------------- #
    #                  PLOT THE HISTOGRAM OF PAIRWISE DATA                    #
    # ----------------------------------------------------------------------- #
    def plot_histogram(self, frame=0, datcmp_data_type="C", display=True,
                       save=False, filename="", directory="", num_bins=20):
        """
        Plot the histogram of the DATCMP data.

        For any chosen frame this methods plots a histogram of the DATCMP data
        for all pairwise comparisons with that frame. If no frames are chosen
        then the method plots all pairwise frame comparison data.

        Parameters
        ----------
        frame : int, optional (default=0)
            Number of the frame for which all pairwise comparison data is
            plotted. If not specified then all pairwise frame comparison data
            is shown.
        datcmp_data_type : str, optional (default="C")
            The data type from the DATCMP results that is plotted.
            The input options are:
            1) 'C' - This will return the C value i.e. the max observed patch
            of continuous runs of -1 or 1
            2) 'P(>C)' - This will return the P value of observing a patch of
            continuous runs of -1 or 1 bigger than the corresponding C value.
            3) 'adj P(>C)' - This will return the Bonferroni adjusted P value
            of observing a patch of continuous runs of -1 or 1 bigger than the
            corresponding C value.
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.
        num_bins : int, optional (default=20)
            Number of bins used to plot in the histogram.

        Examples
        --------
        Plot histogram of all of the DATCMP data with data type "C" (as defined
        above)

        >>>  scat_obj.plot_histogram()

        Plot histogram of all of Bonferroni corrected P values with all frames
        compared to frame 10.

        >>> scat_obj.plot_histogram(frame=10, datcmp_data_type="adj P(>C)")

        Save the histogram in the current working directory without displaying
        it

        >>>  scat_obj.plot_histogram(display=False, save=True, filename="MyPlot.png")
        """
        if datcmp_data_type == "C" or datcmp_data_type == 0:
            dat_type = 0
        elif datcmp_data_type == "P(>C)":
            dat_type = 1
        elif datcmp_data_type == "adj P(>C)":
            dat_type = 2
        else:
            print "********************** ERROR ***************************"
            print "INVALID DATCMP DATA TYPE CHOSEN: '{}' DOES NOT EXIST".format(datcmp_data_type)
            print "Please choose either 'C', 'P(>C)' or 'adj P(>C)'."
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          PLOT HISTOGRAM                         #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        plt.figure(self.PLOT_NUM)
        plt.hist(self.get_pw_data_array(frame)[:, dat_type], bins=num_bins)
        plt.xlabel("{}".format(datcmp_data_type), fontdict=self.PLOT_LABEL)
        plt.ylabel(r'Frequency', fontdict=self.PLOT_LABEL)
        if frame == 0:
            plt.title("{} values for all pairwise comparisons".format(datcmp_data_type))
        else:
            plt.title("{} values for all pairwise comparisons with frame {}".format(datcmp_data_type, frame))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

    # ----------------------------------------------------------------------- #
    #                    SCATTER PLOT OF PAIRWISE DATA                        #
    # ----------------------------------------------------------------------- #
    def plot_scatter(self, frame=1, P_threshold=0.01, markersize=60,
                     display=True, save=False, filename="", directory="",
                     legend_loc="upper left", x_change=False, use_adjP=True,
                     xaxis_frame_num=False,
                     colours=["#0072B2", "#009E73", "#D55E00"],
                     markers=["o", "o", "o"]):
        """
        Create scatter plot of the frame similarity data.

        For any chosen frame this methods creates a scatter plot of the C
        values - the maximum observed patch of continuous runs of -1 or 1 - for
        a chosen frame against all other frames. The probabilities of those C
        values occuring by chance are also coloured accordingly, and this can
        be changed by the user.

        Parameters
        ----------
        frame : int, optional (default=0)
            Number of the frame for which all pairwise comparison data is
            plotted. If not specified then all pairwise frame comparison data
            against frame 1 is plotted.
        P_threshold : float, optional (default=0.01)
            The significance level for the data. If the probability is below
            this value then the current frame is considered dissimilar to the
            frame to which it is being compared. Default significance level is
            0.01.
        markersize : float, optional (default=60)
            Change the size of the markers on the plot. Default value is 60.
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.
        legend_loc : str or int, optional (default="upper left")
            Set the location of the legend in the figure. All possible values
            can be found in the Matplotlib Legend API documentation:
            http://matplotlib.org/api/legend_api.html. Default is set to the
            upper left of the figure.
        x_change : bool, optional (default=False)
            This option changes whether the x-axis is plotted with absolute
            values (x_change=True) or if it's plotted with differences between
            the x-axis values (x_change=False). For example if we compare
            choose the reference frame to be frame 10 (frame=10) then compare
            this with frame 30 then setting x_change=True will put the relevant
            point at frame 30. However if we set x_change=False this will put
            the relevant point to be at abs(10 - 30) = 20. This option allows
            you to determine if the dissimilar frames are frames that are
            collected far from the current frame. Default is set to
            x_change=True.
        use_adjP : bool, optional (default=True)
            Choose whether to use the Bonferroni corrected P values or to use
            the unadjusted P values. Default is to use the Bonferroni corrected
            P values.
        xaxis_frame_num : bool, optional (default=True)
            Choose whether to plot the x-axis with the frame number or whether
            to use the custom x-axis metric values that may (or may not) have
            been defined when the ScatterAnalysis object was constructed.
            Default is to use the custom metric (if it was defined).
        colours : list, optional (default=["#0072B2", "#009E73", "#D55E00"])
            Choose the colour scheme for the markers in the scatter plot, which
            is differentiated by the P value.
        markers : list, optional (default=["o", "o", "o"])
            Choose the markers used for the markers in the scatter plot, which
            is differentiated by the P value. A list of the markers that can be
            used can be found on the Matplotlib Markers API webpage:
            http://matplotlib.org/api/markers_api.html


        Examples
        --------
        Create the scatter plot of the data compared against frame 1

        >>>  scat_obj.plot_scatter()

        Create scatter plot where all frames are compared against frame 30 and
        the x axis plots the difference in frame number from frame 30.

        >>> scat_obj.plot_scatter(frame=30, x_change=True)

        Save the scatter plot in the current working directory without
        displaying it.

        >>>  scat_obj.plot_scatter(display=False, save=True, filename="MyPlot.png")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SCATTER PLOT                           #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        pwframe_data = self.get_pw_data_array(frame=frame,
                                              delete_zero_row=False)

        if xaxis_frame_num:
            x_axis = np.linspace(1, self.I.shape[1], self.I.shape[1])
        else:
            x_axis = self.x_axis

        if x_change:
            sub = x_axis[frame - 1]
            pwframe_data = np.column_stack([abs(x_axis - sub), pwframe_data])
        else:
            pwframe_data = np.column_stack([x_axis, pwframe_data])
        pwframe_data = np.delete(pwframe_data, (frame-1), axis=0)

        if use_adjP:
            lb_dict = {0: [colours[0], "P(>C) = 1"],
                       1: [colours[1], "{} <= P(>C) < 1".format(P_threshold)],
                       2: [colours[2], "P(>C) < {}".format(P_threshold)]}
            P_col = 3
        else:
            lb_dict = {0: [colours[0], "P(>C) == 1"],
                       1: [colours[1], "{} <= P(>C) < 1".format(P_threshold)],
                       2: [colours[2], "P(>C) < {}".format(P_threshold)]}
            P_col = 2

        plt.figure(self.PLOT_NUM)
        good_points = pwframe_data[pwframe_data[:, P_col] == 1]
        plt.scatter(good_points[:, 0], good_points[:, 1], color=lb_dict[0][0],
                    s=markersize, edgecolors='#ffffff', alpha=1,
                    marker=markers[0], label=lb_dict[0][1])

        ok_points = pwframe_data[1 > pwframe_data[:, P_col]]
        ok_points = ok_points[ok_points[:, P_col] >= P_threshold]
        plt.scatter(ok_points[:, 0], ok_points[:, 1], color=lb_dict[1][0],
                    s=markersize, edgecolors='#ffffff', alpha=1,
                    marker=markers[1], label=lb_dict[1][1])

        bad_points = pwframe_data[pwframe_data[:, P_col] < P_threshold]
        plt.scatter(bad_points[:, 0], bad_points[:, 1], color=lb_dict[2][0],
                    s=markersize, edgecolors='#ffffff', alpha=1,
                    marker=markers[2], label=lb_dict[2][1])

        plt.legend(loc=legend_loc, scatterpoints=1)
        if x_change:
            if xaxis_frame_num:
                plt.xlabel(r'$\Delta${}'.format("Frame Number"),
                               fontdict=self.PLOT_LABEL)
            elif self.x_units:
                plt.xlabel(r'$\Delta${} ({})'.format(self.x_metric, self.x_units),
                               fontdict=self.PLOT_LABEL)
            else:
                plt.xlabel(r'$\Delta${}'.format(self.x_metric),
                               fontdict=self.PLOT_LABEL)
        else:
            if xaxis_frame_num:
                plt.xlabel("Frame Number", fontdict=self.PLOT_LABEL)
            elif self.x_units:
                plt.xlabel("{} ({})".format(self.x_metric, self.x_units),
                               fontdict=self.PLOT_LABEL)
            else:
                plt.xlabel("{}".format(self.x_metric),
                               fontdict=self.PLOT_LABEL)
        plt.ylabel(r'C', fontdict=self.PLOT_LABEL)
        # if xaxis_frame_num:
        #     ax1.set_title("C values against frame number for frame {}".format(frame))
        # else:
        #     ax1.set_title("C values against {} for frame {}".format(self.x_metric,
        #                                                         frame))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

    # ----------------------------------------------------------------------- #
    #                           P(>C) HEAT MAP                                #
    # ----------------------------------------------------------------------- #
    def plot_heatmap(self, P_threshold=0.01, markersize=60,
                     display=True, save=False, filename="", directory="",
                     legend_loc=2, x_change=False, use_adjP=True,
                     xaxis_frame_num=False, P_values=True,
                     colours=["#0072B2", "#009E73", "#D55E00"]):
        """
        Create Heatmap of the frame similarity data.

        The heatmap provides a visual representation of the pairwise similarity
        analyis of the SAXS dataset.

        Parameters
        ----------
        P_threshold : float, optional (default=0.01)
            The significance level for the data. If the probability is below
            this value then one frame is considered dissimilar to the
            frame to which it is being compared. Default significance level is
            0.01.
        markersize : float, optional (default=60)
            Change the size of the markers on the plot. Default value is 60.
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.
        legend_loc : str or int, optional (default=2)
            Set the location of the legend in the figure. All possible values
            can be found in the Matplotlib Legend API documentation:
            http://matplotlib.org/api/legend_api.html. Default is set to the
            upper left (integer value is 2) of the figure.
        x_change : bool, optional (default=False)
            This option changes whether the x-axis is plotted with absolute
            values (x_change=True) or if it's plotted with differences between
            the x-axis values (x_change=False). For example if we compare
            choose the reference frame to be frame 10 (frame=10) then compare
            this with frame 30 then setting x_change=True will put the relevant
            point at frame 30. However if we set x_change=False this will put
            the relevant point to be at abs(10 - 30) = 20. This option allows
            you to determine if the dissimilar frames are frames that are
            collected far from the current frame. Default is set to
            x_change=True.
        use_adjP : bool, optional (default=True)
            Choose whether to use the Bonferroni corrected P values or to use
            the unadjusted P values. Default is to use the Bonferroni corrected
            P values.
        xaxis_frame_num : bool, optional (default=True)
            Choose whether to plot the x-axis with the frame number or whether
            to use the custom x-axis metric values that may (or may not) have
            been defined when the ScatterAnalysis object was constructed.
            Default is to use the custom metric (if it was defined).
        P_values : bool, optional (default=True)
            Choose whether to plot probability values or C values - the maximum
            observed patch of continuous runs of -1 or 1. Default is to use the
            P values.
        colours : list, optional (default=["#0072B2", "#009E73", "#D55E00"])
            Choose the colour scheme for the markers in the scatter plot, which
            is differentiated by the P value.

        Examples
        --------
        Create heatmap.

        >>>  scat_obj.plot_heatmap()

        Save the heatmap in the current working directory withot displaying it.

        >>>  scat_obj.plot_heatmap(display=False, save=True, filename="MyPlot.png")
        """
        full_data = []
        num_frames = self.I.shape[1]
        for frame in range(1, num_frames+1):
            pwframe_data = self.get_pw_data_array(frame=frame,
                                                  delete_zero_row=False)

            if xaxis_frame_num:
                x_axis = np.linspace(1, self.I.shape[1], self.I.shape[1])
            else:
                x_axis = self.x_axis

            if x_change:
                sub = x_axis[frame - 1]
                pwframe_data = np.column_stack([abs(x_axis - sub), pwframe_data])
            else:
                pwframe_data = np.column_stack([x_axis, pwframe_data])
            pwframe_data = np.delete(pwframe_data, (frame-1), axis=0)
            if use_adjP:
                P_col = 3
            else:
                P_col = 2

            good_points = pwframe_data[pwframe_data[:, P_col] == 1]
            ok_points = pwframe_data[1 > pwframe_data[:, P_col]]
            ok_points = ok_points[ok_points[:, P_col] >= P_threshold]
            bad_points = pwframe_data[pwframe_data[:, P_col] < P_threshold]

            xOrder = list(good_points[:, 0]) + list(ok_points[:, 0]) + list(bad_points[:, 0])
            C_values = list(good_points[:, 1]) + list(ok_points[:, 1]) + list(bad_points[:, 1])
            xData = [-1]*len(good_points[:, 0]) + [0]*len(ok_points[:, 0]) + [1]*len(bad_points[:, 0])

            if P_values:
                xOrder_sorted, xData_sorted = (list(t) for t in zip(*sorted(zip(xOrder, xData))))
                full_data.append(xData_sorted)

            else:
                xOrder_sorted, C_values_sorted = (list(t) for t in zip(*sorted(zip(xOrder, C_values))))
                full_data.append(C_values_sorted)

        full_DataFrame = pd.DataFrame(data=full_data,
                                      columns=xOrder_sorted,
                                      index=range(1, num_frames+1))
        heatmap = plt.figure()
        ax = plt.subplot(111)

        if P_values:
            sns.heatmap(full_DataFrame, cmap=mpl.colors.ListedColormap(colours), cbar=False)

            # create legend information
            if use_adjP:
                good_label = "adj P(>C) == 1"
                ok_label = "{} <= adj P(>C) < 1".format(P_threshold)
                bad_label = "adj P(>C) < {}".format(P_threshold)
                plot_title = "adj P(>C) values for varying frame number"
            else:
                good_label = "P(>C) == 1"
                ok_label = "{} <= P(>C) < 1".format(P_threshold)
                bad_label = "P(>C) < {}".format(P_threshold)
                plot_title = "P(>C) values for varying frame number"
            good_patch = mpl.patches.Patch(color=colours[0], label=good_label)
            ok_patch = mpl.patches.Patch(color=colours[1], label=ok_label)
            bad_patch = mpl.patches.Patch(color=colours[2], label=bad_label)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            lgd = ax.legend(handles=[good_patch, ok_patch, bad_patch], bbox_to_anchor=(1, 1), loc=legend_loc)

        else:
            sns.heatmap(full_DataFrame, cmap="YlGnBu", cbar=True)
            plot_title = "C values for varying frame number"

        if xaxis_frame_num:
            plt.xlabel('Frame Number', fontdict=self.PLOT_LABEL)
        else:
            plt.xlabel('Frame Number', fontdict=self.PLOT_LABEL)
            # if x_change:
            #     if self.x_units:
            #         plt.xlabel(r'$\Delta${} ({})'.format(self.x_metric, self.x_units),
            #                    fontdict=self.PLOT_LABEL)
            #     else:
            #         plt.xlabel(r'$\Delta${}'.format(self.x_metric),
            #                    fontdict=self.PLOT_LABEL)
            # else:
            #     if self.x_units:
            #         plt.xlabel("{} ({})".format(self.x_metric, self.x_units),
            #                    fontdict=self.PLOT_LABEL)
            #     else:
            #         plt.xlabel("{}".format(self.x_metric),
            #                    fontdict=self.PLOT_LABEL)
        plt.ylabel('Frame Number', fontdict=self.PLOT_LABEL)
        plt.title(plot_title)
        tick_labels = np.linspace(1, num_frames, 10).astype(int)
        plt.yticks(tick_labels[::-1], tick_labels)
        plt.xticks(tick_labels, tick_labels, rotation='horizontal')

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            heatmap.show()

    # ----------------------------------------------------------------------- #
    #                        PLOT 1D INTENSITY CURVE                          #
    # ----------------------------------------------------------------------- #
    def plot_1d_intensity(self, frames, start_point=1, end_point=-1,
                          log_intensity=True, display=True, save=False,
                          filename="", directory="", legend_loc="upper right",
                          markersize=8):
        """
        Plot 1D scatter curves.

        Choose frames for which to plot the 1D scatter curves in the same
        figure.

        Parameters
        ----------
        frames : int or list
            Number of the frame or list of frame numbers to be plotted on the
            figure.
        start_point : int, optional (default=1)
            The point on the scatter curve from which you would like to start
            the plot. Default is to plot from the first point.
        end_point : int, optional (default=-1)
            The point on the scatter curve from which you would like to end
            the plot. Default is to plot up until to very last point on the
            curve.
        log_intensity : bool, optional (default=True)
            Choose whether to plot the intensity curve on a logarithmic scale
            (default) or on an absolute scale.
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.
        legend_loc : str or int, optional (default="upper left")
            Set the location of the legend in the figure. All possible values
            can be found in the Matplotlib Legend API documentation:
            http://matplotlib.org/api/legend_api.html. Default is set to the
            upper left of the figure.
        markersize : float, optional (default=60)
            Change the size of the markers on the plot. Default value is 60.


        Examples
        --------
        Plot the 1D intensity curve for frames 1, 10 and 100

        >>>  scat_obj.plot_1d_intensity([1, 10, 100], start_point=200, end_point=400)

        Plot the 1D intensity curve for frames 1, 10 and 100 between point 200
        and 400.

        >>>  scat_obj.plot_1d_intensity([1, 10, 100])

        Save the 1D intensity plot in the current working directory without
        displaying it.

        >>>  scat_obj.plot_1d_intensity(display=False, save=True, filename="MyPlot.png")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                        PLOT INTENSITY CURVE                     #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if end_point == -1:
            end_point = len(self.q)
        reciprocal_resolution = self.q[start_point-1:end_point]
        if isinstance(frames, list):
            frames = [i - 1 for i in frames]
        intensity = self.I[start_point-1:end_point, frames]
        if log_intensity:
            intensity = np.log(intensity)
        plt.figure(self.PLOT_NUM)
        if len(intensity.shape) == 2:
            for i in xrange(0, intensity.shape[1]):
                if self.x_units:
                    plt.plot(reciprocal_resolution, intensity[:, i], 'o',
                             markersize=markersize, markeredgecolor='#ffffff',
                             markeredgewidth=0.2,
                             label="Frame {}, {}={:.2f} {}".format(frames[i] + 1,
                                                                   self.x_metric,
                                                                   self.x_axis[frames[i]],
                                                                   self.x_units))
                else:
                    plt.plot(reciprocal_resolution, intensity[:, i], 'o',
                             label="Frame {}".format(frames[i] + 1))
        else:
            if self.x_units:
                plt.plot(reciprocal_resolution, intensity, 'o',
                         label="Frame {}, {}={:.2f} {}".format(frames,
                                                               self.x_metric,
                                                               self.x_axis[frames-1],
                                                               self.x_units))
            else:
                plt.plot(reciprocal_resolution, intensity, 'o',
                         label="Frame {}".format(frames))
        plt.xlabel(r'Scattering Vector, q ($nm^{-1}$)',
                   fontdict=self.PLOT_LABEL)
        if log_intensity:
            plt.ylabel('log(I) (arb. units.)', fontdict=self.PLOT_LABEL)
        else:
            plt.ylabel('Intensity (arb. units.)', fontdict=self.PLOT_LABEL)
        plt.title('1D Scattering Curve', fontdict=self.PLOT_LABEL)
        plt.legend(loc=legend_loc, scatterpoints=1)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

    # ----------------------------------------------------------------------- #
    #              PLOT FIRST N DIFFERENT FRAMES FOR EACH FRAME               #
    # ----------------------------------------------------------------------- #
    def plot_first_n_diff_frames(self, n=1, P_threshold=0.01,
                                 P_type="adjP", display=True, save=False,
                                 filename="", directory=""):
        """
        Plot dissimilar frames

        For each frame in the dataset you can perform a similarity analysis
        against every other frame. From this you can calculate the first frame
        for which the probability value (P(>C) or adj P(>C)) for n consecutive
        frames is below the significance value (signifying that the frames are
        likely to be dissimilar). Note here that n and the P_threshold
        (significance value) are user defined values.
        This methods plots the first frame in that consecutive run for each
        'reference' frame in the dataset. By reference frame, we mean the frame
        for which every other frame is compared.

        Parameters
        ----------
        n : int, optional (default=1)
            Number of consecutive dissimilar frames to be considered
            dissimilar.
        P_threshold : float, optional (default=0.01)
            The significance level for the data. If the probability is below
            this value then one frame is considered dissimilar to the
            frame to which it is being compared. Default significance level is
            0.01.
        P_type : str, optional (default="adjP")
            String denoting whether to use the Bonferroni corrected P value
            (input string="adjP") or the ordinary P value (input string="P").
            Default is to use the Bonferroni corrected P value.
        display : bool, optional (default=True)
            Choose whether to display the plot. By default it is displayed.
        save : bool, optional (default=True)
            Choose whether to save the plot. By default the plot is not saved.
        filename : str, optional  (default="")
            Choose a filename for the plot that you want to save. This includes
            the file extension.
        directory : str, optional  (default="")
            Select the directory in which you want the plot to be saved.

        Examples
        --------
        Plot the first dissimilar frame for each frame in the dataset used as
        the reference frame.

        >>>  scat_obj.plot_first_n_diff_frames()

        Plot the first of 3 consecutive dissimilar frames for each frame in the
        dataset used as the reference frame (as was the case described in
        Brooks-Bartlett et al. 2016 - Development of tools to automate
        quantitative analysis of radiation damage in SAXS experiments).

        >>>  scat_obj.plot_first_n_diff_frames(n=3)

        Save the plot in the current working directory without displaying it.

        >>>  scat_obj.plot_first_n_diff_frames(display=False, save=True, filename="MyPlot.png")
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                          SET PLOT PARAMS                        #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.PLOT_NUM += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                            PLOT CURVE                           #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        num_frames = self.I.shape[1]
        diff_frames_list = np.zeros(num_frames)
        frames = np.linspace(1, num_frames, num_frames)
        for i in xrange(0, num_frames):
            frame = i+1
            diff_frames_list[i] = self.find_first_n_diff_frames(n=n, frame=frame, P_threshold=P_threshold, P_type=P_type)

        plt.figure(self.PLOT_NUM)
        plt.plot(frames, diff_frames_list)
        plt.xlabel("Reference Frame Number")
        if n == 1:
            plt.ylabel("First Dissimilar Frame")
        else:
            plt.ylabel("First of {} Consecutive Dissimilar Frames".format(n))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        #                       SAVE AND/OR DISPLAY                       #
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if save and filename:
            if directory:
                plot_path = "{}/{}".format(directory, filename)
            else:
                plot_path = filename
            plt.savefig(plot_path)
        elif save and not filename:
            print "********************** ERROR ***************************"
            print "COULD NOT SAVE PLOT"
            print "No filename specified. Please specify a filename if you"
            print "would like to save the plot."
        if display:
            plt.show()

# --------------------------------------------------------------------------- #
#                               FUNCTIONS                                     #
# --------------------------------------------------------------------------- #


def run_system_command(command_string):
    """Function used to run the system command and return the log"""
    process = subprocess.Popen(command_string, stdout=subprocess.PIPE,
                               shell=True)  # Run system command
    output = process.communicate()  # Get the log.
    return output[0]  # return the log file


def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles


def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5  # This varies with the font!!
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)


def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001
    if cos(rotation) > threshold:
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold:
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold:
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold:
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)
