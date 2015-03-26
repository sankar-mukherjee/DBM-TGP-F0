#!/usr/bin/perl
# MLSA
# Put .mgc files in gen folder along with their aschii pitch files
# Sankar Mukherjee
# CET IITKGP

# Settings ==============================

%ordr = ('mgc' => '35',     # feature order
         'lf0' => '1',
         'dur' => '5');

# Speech Analysis/Synthesis Setting ==============
# speech analysis
$sr = 44100;   # sampling rate (Hz)
$fs = 220; # frame period (point)
$fw = 0.55;   # frequency warping
$gm = 0;      # pole/zero representation weight
$lg = 1;     # use log gain instead of linear gain
$fr = $fs/$sr;      # frame period (sec)

# speech synthesis
$pf = 1.4; # postfiltering factor
$fl = 4096;    # length of impulse response
$co = 2047;        # order of cepstrum to approximate mel-generalized cepstrum


# Modeling/Generation Setting ==============
# generation
$maxEMiter  = 20;  # max EM iteration
$EMepsilon  = 0.0001;  # convergence factor for EM iteration
$useGV      = 1;      # turn on GV
$maxGViter  = 50;  # max GV iteration
$GVepsilon  = 0.0001;  # convergence factor for GV iteration
$minEucNorm = 0.01; # minimum Euclid norm for GV iteration
$stepInit   = 1.0;   # initial step size
$stepInc    = 1.2;    # step size acceleration factor
$stepDec    = 0.5;    # step size deceleration factor
$hmmWeight  = 1.0;  # weight for HMM output prob.
$gvWeight   = 1.0;   # weight for GV output prob.
$optKind    = 'NEWTON';  # optimization method (STEEPEST, NEWTON, or LBFGS)
$nosilgv    = 1;    # GV without silent and pause phoneme
$cdgv       = 1;       # context-dependent GV


# Directories & Commands ===============
# project directories
use Cwd;

$prjdir = cwd();
$datdir = "$prjdir/data";
# Perl
$PERL = '/usr/bin/perl';

# wc
$WC = '/usr/bin/wc';

# tee
$TEE = '/usr/bin/tee';

# HTS commands
$HCOMPV    = '/usr/local/HTS-2.2beta/bin/HCompV';
$HLIST     = '/usr/local/HTS-2.2beta/bin/HList';
$HINIT     = '/usr/local/HTS-2.2beta/bin/HInit';
$HREST     = '/usr/local/HTS-2.2beta/bin/HRest';
$HEREST    = '/usr/local/HTS-2.2beta/bin/HERest';
$HHED      = '/usr/local/HTS-2.2beta/bin/HHEd';
$HSMMALIGN = '/usr/local/HTS-2.2beta/bin/HSMMAlign';
$HMGENS    = '/usr/local/HTS-2.2beta/bin/HMGenS';
$ENGINE    = '/usr/local/bin/hts_engine';

# SPTK commands
$X2X      = '/usr/local/SPTK/bin/x2x';
$FREQT    = '/usr/local/SPTK/bin/freqt';
$C2ACR    = '/usr/local/SPTK/bin/c2acr';
$VOPR     = '/usr/local/SPTK/bin/vopr';
$MC2B     = '/usr/local/SPTK/bin/mc2b';
$SOPR     = '/usr/local/SPTK/bin/sopr';
$B2MC     = '/usr/local/SPTK/bin/b2mc';
$EXCITE   = '/usr/local/SPTK/bin/excite';
$LSP2LPC  = '/usr/local/SPTK/bin/lsp2lpc';
$MGC2MGC  = '/usr/local/SPTK/bin/mgc2mgc';
$MGLSADF  = '/usr/local/SPTK/bin/mglsadf';
$MERGE    = '/usr/local/SPTK/bin/merge';
$BCP      = '/usr/local/SPTK/bin/bcp';
$LSPCHECK = '/usr/local/SPTK/bin/lspcheck';
$BCUT     = '/usr/local/SPTK/bin/bcut';
$VSTAT    = '/usr/local/SPTK/bin/vstat';
$NAN      = '/usr/local/SPTK/bin/nan';
$DFS      = '/usr/local/SPTK/bin/dfs';
$SWAB     = '/usr/local/SPTK/bin/swab';

# SoX (to add RIFF header)
$SOX       = '/usr/bin/sox';
$SOXOPTION = '2';

#==============================================Main Program=====================================================
$line  = `ls gen/*.mgc`;
@FILE  = split( '\n', $line );
foreach $file (@FILE) {
	$base = `basename $file .mgc`;
      	chomp($base);
	$line = "$X2X +af gen/$base > gen/$base.pit";		#change to float value for the .mgs name file pitch
	shell($line);
}
gen_wave("$prjdir/gen");

#========================================================================================================

# sub routine for speech synthesis from log f0 and Mel-cepstral coefficients
sub gen_wave($) {
   my ($gendir) = @_;
   my ( $line, @FILE, $file, $base );

   $line  = `ls $gendir/*.mgc`;
   @FILE  = split( '\n', $line );
   $lgopt = "-l" if ($lg);

   print "Processing directory $gendir:\n";
   foreach $file (@FILE) {
      $base = `basename $file .mgc`;
      chomp($base);
      if ( -s $file && -s "$gendir/$base.pit" ) {
         print " Synthesizing a speech waveform from $base.mgc and $base.lf0...";

         # convert log F0 to pitch
         #lf02pitch( $base, $gendir );

         if ( $gm > 0 ) {

            # MGC-LSPs -> MGC coefficients
            $line = "$LSPCHECK -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " -r 0.1 $file | ";
            $line .= "$LSP2LPC -m " . ( $ordr{'mgc'} - 1 ) . " -s " . ( $sr / 1000 ) . " $lgopt | ";
            $line .= "$MGC2MGC -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -c $gm -n -u -M " . ( $ordr{'mgc'} - 1 ) . " -A $fw -C $gm " . " > $gendir/$base.c_mgc";
            shell($line);

            $mgc = "$gendir/$base.c_mgc";
         }
         else {

            # apply postfiltering
            if ( $gm == 0 && $pf != 1.0 && $useGV == 0 ) {
               postfiltering( $base, $gendir );
               $mgc = "$gendir/$base.p_mgc";
            }
            else {
               $mgc = $file;
            }
         }

         # synthesize waveform
         $lfil = `$PERL $datdir/scripts/makefilter.pl $sr 0`;
         $hfil = `$PERL $datdir/scripts/makefilter.pl $sr 1`;

         $line = "$SOPR -m 0 $gendir/$base.pit | $EXCITE -p $fs | $DFS -b $hfil > $gendir/$base.unv";
         shell($line);

         $line = "$EXCITE -p $fs $gendir/$base.pit | ";
         $line .= "$DFS -b $lfil | $VOPR -a $gendir/$base.unv | ";
         $line .= "$MGLSADF -m " . ( $ordr{'mgc'} - 1 ) . " -p $fs -a $fw -c $gm $mgc | ";
         $line .= "$X2X +fs -o | ";
         $line .= "$SOX -c 1 -s -$SOXOPTION -t raw -r $sr - -c 1 -s -$SOXOPTION -t wav -r $sr $gendir/$base.wav";
         shell($line);

         $line = "rm -f $gendir/$base.unv";
         shell($line);

         print "done\n";
      }
   }
   print "done\n";
}

# sub routine for log f0 -> f0 conversion
sub lf02pitch($$) {
   my ( $base, $gendir ) = @_;
   my ( $t, $T, $data );

   # read log f0 file
   open( IN, "$gendir/${base}.lf0" );
   @STAT = stat(IN);
   read( IN, $data, $STAT[7] );
   close(IN);

   # log f0 -> pitch conversion
   $T = $STAT[7] / 4;
   @frq = unpack( "f$T", $data );
   for ( $t = 0 ; $t < $T ; $t++ ) {
      if ( $frq[$t] == -1.0e+10 ) {
         $out[$t] = 0.0;
      }
      else {
         $out[$t] = $sr / exp( $frq[$t] );
      }
   }
   $data = pack( "f$T", @out );

   # output data
   open( OUT, ">$gendir/${base}.pit" );
   print OUT $data;
   close(OUT);
}

# sub routine for formant emphasis in Mel-cepstral domain
sub postfiltering($$) {
   my ( $base, $gendir ) = @_;
   my ( $i, $line );

   # output postfiltering weight coefficient
   $line = "echo 1 1 ";
   for ( $i = 2 ; $i < $ordr{'mgc'} ; $i++ ) {
      $line .= "$pf ";
   }
   $line .= "| $X2X +af > $gendir/weight";
   shell($line);

   # calculate auto-correlation of original mcep
   $line = "$FREQT -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -M $co -A 0 < $gendir/${base}.mgc | ";
   $line .= "$C2ACR -m $co -M 0 -l $fl > $gendir/${base}.r0";
   shell($line);

   # calculate auto-correlation of postfiltered mcep
   $line = "$VOPR -m -n " . ( $ordr{'mgc'} - 1 ) . " < $gendir/${base}.mgc $gendir/weight | ";
   $line .= "$FREQT -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw -M $co -A 0 | ";
   $line .= "$C2ACR -m $co -M 0 -l $fl > $gendir/${base}.p_r0";
   shell($line);

   # calculate MLSA coefficients from postfiltered mcep
   $line = "$VOPR -m -n " . ( $ordr{'mgc'} - 1 ) . " < $gendir/${base}.mgc $gendir/weight | ";
   $line .= "$MC2B -m " . ( $ordr{'mgc'} - 1 ) . " -a $fw | ";
   $line .= "$BCP -n " .  ( $ordr{'mgc'} - 1 ) . " -s 0 -e 0 > $gendir/${base}.b0";
   shell($line);

   # calculate 0.5 * log(acr_orig/acr_post)) and add it to 0th MLSA coefficient
   $line = "$VOPR -d < $gendir/${base}.r0 $gendir/${base}.p_r0 | ";
   $line .= "$SOPR -LN -d 2 | ";
   $line .= "$VOPR -a $gendir/${base}.b0 > $gendir/${base}.p_b0";
   shell($line);

   # generate postfiltered mcep
   $line = "$VOPR -m -n " . ( $ordr{'mgc'} - 1 ) . " < $gendir/${base}.mgc $gendir/weight | ";
   $line .= "$MC2B -m " .  ( $ordr{'mgc'} - 1 ) . " -a $fw | ";
   $line .= "$BCP -n " .   ( $ordr{'mgc'} - 1 ) . " -s 1 -e " . ( $ordr{'mgc'} - 1 ) . " | ";
   $line .= "$MERGE -n " . ( $ordr{'mgc'} - 2 ) . " -s 0 -N 0 $gendir/${base}.p_b0 | ";
   $line .= "$B2MC -m " .  ( $ordr{'mgc'} - 1 ) . " -a $fw > $gendir/${base}.p_mgc";
   shell($line);
}

sub shell($) {
   my ($command) = @_;
   my ($exit);

   $exit = system($command);

   if ( $exit / 256 != 0 ) {
      die "Error in $command\n";
   }
}
