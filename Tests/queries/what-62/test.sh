
# run this file from the CS1003-P4 directory
# by running: Tests/queries/what-62/test.sh

export CLASSPATH=${CLASSPATH}:"/cs/studres/CS1003/0-General/spark/*":.

java CS1003P4 Tests/data "what sort of madness is this" 0.62 > Tests/queries/what-62/stdout 2> Tests/queries/what-62/stderr

LC_ALL=C sort Tests/queries/what-62/stdout -o Tests/queries/what-62/stdout

diff -y Tests/queries/what-62/stdout Tests/queries/what-62/expected.txt
