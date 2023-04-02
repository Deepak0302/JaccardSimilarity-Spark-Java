
# run this file from the CS1003-P4 directory
# by running: Tests/queries/advice/test.sh

export CLASSPATH=${CLASSPATH}:"/cs/studres/CS1003/0-General/spark/*":.

java CS1003P4 Tests/data "she generally gave herself very good advice though she very seldom followed it" 0.95 > Tests/queries/advice/stdout 2> Tests/queries/advice/stderr

LC_ALL=C sort Tests/queries/advice/stdout -o Tests/queries/advice/stdout

# diff Tests/queries/advice/stdout Tests/queries/advice/expected.txt
