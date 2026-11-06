# Recorded failure analysis

The learned head makes 14 mistakes out of 300 assessment images; the mapped pretrained classifier makes 8. This fails the reason to replace the simpler baseline. Keep the head as an experiment and the original pretrained model as the service default.

Gas-pump recall is 26/30 on clean inputs and 11/30 after the fixed radius-2 blur; church recall drops from 29/30 to 18/30. Golf-ball recall remains 28/30 in both conditions. These results suggest different reliance on fine image detail, but the experiment does not establish a causal explanation for individual errors. Each class contains only 30 assessment images.

The report retains every clean head error by descending confidence. For example, `n03888257_3651.JPEG` (parachute) is predicted as chain saw with score 0.9955, while the mapped pretrained model is correct. `ILSVRC2012_val_00020502.JPEG` (golf ball) is predicted as gas pump with score 0.9753, again losing to the baseline. High scores are not evidence that an output is trustworthy. No relabeling was performed to improve the results.

Aggregate clean ECE is approximately 0.030 with ten fixed-width bins, yet these severe high-confidence errors remain. This is a small, balanced, closed-set sample and ECE alone cannot justify deployment calibration. The training head has no rejection class, open-set threshold, OOD detector, camera adaptation or drift monitor.

The CPU load experiment also does not show a uniform batching gain. A 5 ms wait adds work latency for a lone request; varying concurrency changes queueing, decode contention and CPU execution. The short shared-laptop runs differ even at batch cap 1, where a batching wait cannot collect more images. Treat those differences as measurement variability, retain the complete matrix, and run repeated isolated GPU experiments before choosing defaults.
