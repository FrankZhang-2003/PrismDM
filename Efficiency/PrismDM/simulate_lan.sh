#!/bin/bash

# Set the network interface (using lo, the local loopback interface)
DEV=lo

# Delete existing rules
if [ "$1" == "del" ]
then
    tc qdisc del dev $DEV root
    echo "Traffic control rules for $DEV interface have been deleted."
    exit 0
fi

# Simulate LAN (Local Area Network) effect
if [ "$1" == "lan" ]
then
    # Delete existing rules
    tc qdisc del dev $DEV root 2>/dev/null

    # Set bandwidth to 1 Gbps
    tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 latency 0.1ms

    # Set one-way latency to 0.05 milliseconds (RTT of 0.1 milliseconds)
    tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.05msec

    echo "$DEV interface has been configured for LAN effect: 1 Gbps bandwidth, 0.1 ms RTT."
    exit 0
fi

# If no valid argument is provided, display help information
echo "Usage:"
echo "  $0 lan        # Configure LAN effect (1 Gbps bandwidth, 0.1 ms RTT)"
echo "  $0 del        # Delete traffic control rules"
exit 1
