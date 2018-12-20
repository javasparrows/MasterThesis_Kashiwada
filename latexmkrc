$latex = 'uplatex';
$bibtex = 'upbibtex';
$dvipdf = 'dvipdfmx %O -o %D %S';
$makeindex = 'mendex -U %O -o %D %S';
$pdf_mode = 3; 

$ENV{'TEXINPUTS'}='./sty//:' . $ENV{'TEXINPUTS'};
$ENV{'TEXINPUTS'}='./bib//:' . $ENV{'TEXINPUTS'};
$ENV{'TEXINPUTS'}='./bst//:' . $ENV{'TEXINPUTS'};