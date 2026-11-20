test_that("crossprod_rs equals crossprod", {
  set.seed(1)
  for (d in list(c(1, 1, 1), c(500, 31, 125), c(3000, 40, 7), c(10, 3, 0), c(0, 4, 5))) {
    x <- matrix(rnorm(d[1] * d[2]), d[1], d[2]); y <- matrix(rnorm(d[1] * d[3]), d[1], d[3])
    expect_equal(crossprod_rs(x, y), unname(crossprod(x, y)), tolerance = 1e-12)
    expect_equal(crossprod_rs(x, y, threads = 0), unname(crossprod(x, y)), tolerance = 1e-12)
  }
  v <- rnorm(200); y <- matrix(rnorm(2000), 200)
  expect_equal(crossprod_rs(v, y), unname(crossprod(v, y)), tolerance = 1e-12)
  expect_equal(crossprod_rs(matrix(1:6, 3), matrix(1:9, 3)), unname(crossprod(matrix(1:6, 3), matrix(1:9, 3))))
  expect_error(crossprod_rs(matrix(1, 3, 2), matrix(1, 4, 2)), "non-conformable")
})

test_that("matprod_rs equals %*%", {
  set.seed(2)
  for (d in list(c(1, 1, 1), c(500, 31, 125), c(3000, 40, 7), c(10, 3, 0), c(0, 4, 5))) {
    x <- matrix(rnorm(d[2] * d[1]), d[2], d[1]); y <- matrix(rnorm(d[1] * d[3]), d[1], d[3])
    expect_equal(matprod_rs(x, y), unname(x %*% y), tolerance = 1e-12)
  }
  expect_error(matprod_rs(matrix(1, 3, 2), matrix(1, 3, 2)), "non-conformable")
})

