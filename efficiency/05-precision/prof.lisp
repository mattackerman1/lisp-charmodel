(require :sb-sprof)
(load "gpt-eff.lisp")
(sb-sprof:with-profiling (:mode :cpu :sample-interval 0.002 :max-samples 400000 :threads :all :report :flat) (gpt::main))
