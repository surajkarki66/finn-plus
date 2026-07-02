#include <functional>
#include <array>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "static_mux_tb_top.h"


/**
 * Check results. Return true if all results are correct.
 */
template<typename T>
bool check_results(std::vector<T> &expected, std::vector<T> &received) {
    if (expected.size() != received.size()) {
        std::cout << "Expected " << expected.size() << " elements, but received " << received.size() << std::endl;
        return false;
    }
    bool success = true;
    for (unsigned int i =  0; i < expected.size(); ++i) {
        if (expected[i] != received[i]) {
            std::cout << "[ " << i << " ] Expected: " << expected[i] << ". Got: " << received[i] << ". " << std::endl;
            success = false;
        }
    }
    if (!success) {
        std::cout << "EXPECTED: ";
        for (auto& val : expected) { std::cout << val << " ";}
        std::cout << "\nRECEIVED: ";
        for (auto& val : received) { std::cout << val << " ";}
        std::cout << std::endl;
    }
    std::cout << "\t" << (success ? "SUCCESS" : "FAIL") << std::endl;
    return success;
}

/**
 * Simply pass streams and check that they arrive in the correct order. If a stream has no data, it is skipped.
 **/
bool verify_normal() {
    std::cout << "TEST: Normal execution." << std::endl;
    std::vector<T3> expected;
    std::vector<T3> received;
    S1 a;
    S2 b, d, e;
    S3 c, out;

    // Check that empty streams are skipped
    // Execute Mux twice, since we have a leftover value in stream a
    a.write(1); a.write(1); b.write(2); c.write(3); d.write(4);
    expected.insert(expected.end(), {1, 2, 3, 4, 1});
    Mux(a, b, c, d, e, out);
    Mux(a, b, c, d, e, out);
    for (unsigned int i = 0; i < 5; ++i) {
        received.push_back(out.read());
    }

    // Check normal operation
    a.write(1); b.write(2); c.write(3); d.write(4); e.write(5);
    expected.insert(expected.end(), {1, 2, 3, 4, 5});
    Mux(a, b, c, d, e, out);
    for (unsigned int i = 0; i < 5; i++) {
        received.push_back(out.read());
    }

    // Check results
    return check_results(expected, received);
}

/**
 * Two arguments are the same stream
 **/
bool verify_preference() {
    std::cout << "TEST: Multiple same streams." << std::endl;
    std::vector<T3> expected;
    std::vector<T3> received;
    S1 a;
    S2 b, d;
    S3 c, out;

    // At first, since b does not have 2 values, it will be skipped,
    // the second Mux call does not do anything
    a.write(1); b.write(2); c.write(3); d.write(4);
    expected.insert(expected.end(), {1, 2, 3, 4});
    Mux(a, b, c, d, b, out);
    Mux(a, b, c, d, b, out);
    for (unsigned int i = 0; i < 4; ++i) {
        received.push_back(out.read());
    }

    // Now we expect to read b much more often. After the second Mux call,
    // the component would have read 4x from b already, so no "2"s are left.
    a.write(1); b.write(2); c.write(3); d.write(4);
    a.write(1); b.write(2); c.write(3); d.write(4);
    a.write(1); b.write(2); c.write(3); d.write(4);
    expected.insert(expected.end(), {1, 2, 3, 4, 2});
    expected.insert(expected.end(), {1, 2, 3, 4});
    expected.insert(expected.end(), {1, 3, 4});
    Mux(a, b, c, d, b, out);
    Mux(a, b, c, d, b, out);
    Mux(a, b, c, d, b, out);
    for (unsigned int i = 0; i < 12; ++i) {
        received.push_back(out.read());
    }

    // Check results
    return check_results(expected, received);
}

const std::initializer_list<std::function<bool()>> test_functions = { verify_normal, verify_preference };

int main() {
    return !std::all_of(test_functions.begin(), test_functions.end(), [](auto f){ return f(); });
}
