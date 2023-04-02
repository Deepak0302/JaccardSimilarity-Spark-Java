
# run this file from the CS1003-P4 directory
# by running: Tests/queries/sail/test.sh

export CLASSPATH=${CLASSPATH}:"/cs/studres/CS1003/0-General/spark/*":.

java CS1003P4 Tests/data "setting sail to the rising wind" 0.75 > Tests/queries/sail/stdout 2> Tests/queries/sail/stderr

LC_ALL=C sort Tests/queries/sail/stdout -o Tests/queries/sail/stdout

diff -y Tests/queries/sail/stdout Tests/queries/sail/expected.txt
