// Set graph parameters
var margin = {top: 40, right: 20, bottom: 20, left: 40};
var w = 500 - margin.right - margin.left;
var h = 500 - margin.top - margin.bottom;
var padding = 300;

// Set number of clusters in k-means algorithm
k = 3;

// Transition times
const drawInitCentroids = 750;
const drawClusterAssignmentLines = 750;
const grayClusterAssignmentLines = 1000;
const fadeClusterAssignmentLines = 500;
const timeClusterAssignmentLines = drawClusterAssignmentLines + grayClusterAssignmentLines + fadeClusterAssignmentLines;
const waitClusterAssignment = drawInitCentroids + drawClusterAssignmentLines;
const colorClusterAssignment = grayClusterAssignmentLines;
const waitCentroidAssignmentLines = drawInitCentroids + timeClusterAssignmentLines;
const moveCentroidAssignmentLines = 250;
const drawCentroidAssignmentLines = 250;
const waitCentroidShift = waitCentroidAssignmentLines + moveCentroidAssignmentLines + drawCentroidAssignmentLines;
const moveCentroidShift = 250;
const waitClusterDecompose = waitCentroidShift + moveCentroidShift;
const fillCentroidShift = 250;
const updateTime = waitClusterDecompose + fillCentroidShift;

// rowConverter parses data strings as floats
var rowConverter = function(d) {
  return [
    parseFloat(d.x),
    parseFloat(d.y)
  ];
};

// set categorical color scale
var colors = d3.scaleOrdinal(d3.schemeCategory10);

// Select random indices from an array - use for random centroid initializatoion.
var indexSampler = function(array,size) {
                   var results = [],
                   sampled = {};
                   while(results.length<size && results.length<array.length) {
                     const index = Math.trunc(Math.random() * array.length);
                     if(!sampled[index]) {
                       results.push(array[index]);
                       sampled[index] = true;
                     }
                   }
                     return results;
                   };

// Squared error function - computes squared L2 distance between points
var squaredError = function(point1, point2) {
  var sqDifferences = [];
  for (var i = 0; i < point1.length; i++) {
    sqDifferences.push((point1[i] - point2[i])**2);
  };
  return sqDifferences.reduce((a, b) => a + b, 0);
};
// Compute squared error from point to each of centroids
var centroidSE = function(point, centroids) {
  var centroidDistances = [];
  for (var i = 0; i < centroids.length; i++) {
    centroidDistances.push( squaredError(point, centroids[i]) );
  };
  return centroidDistances;
};
// Return the index of the centroid nearest to point
var nearestCentroid = function(point, centroids) {
  var centroidDistances = centroidSE(point, centroids);
  return centroidDistances.indexOf(d3.min(centroidDistances));
}

// Initialize centroids by picking k data points uniformly at random
var initalizeCentroids = function(points, k) {
  var initIndex = indexSampler(d3.range(points.length), k);
  // Deterministic k = 3 centroid indices for testing
  //var initIndex = [75, 225, 375];

  var initCentroids = [];
  for(var i = 0; i < k; i++) {
    initCentroids.push(points[initIndex[i]])
  };
  return [initIndex, initCentroids];
};

// Assign point to clusters by nearest centroid
var assignCluster = function(points, centroids, centroidIndexes) {
  // Create array combining data points and assigned centroids
  var assignedPoints = [];
  for(var i = 0; i < points.length; i++) {
    assignedPoints.push([ points[i], centroids[centroidIndexes[i]], centroidIndexes[i] ]);
  };
  return assignedPoints;
};

// Update centroids using current cluster assignment
var centroidUpdate = function(clusteredPoints) {
  var shiftedCentroids = [];
  var clusteredCentroids = [];
  for(var i = 0; i < k; i++){
    // extract the data points by cluster assignment
    var currentCluster = clusteredPoints.filter(entry => entry[2] === i).map(entry => entry[0]);
    // compute the mean of the x-coordinates
    var centroidUpdateX = currentCluster.map(point => point[0]).reduce((total, current) => total += current)/currentCluster.length;
    // compute the mean of the y-coordinates
    var centroidUpdateY = currentCluster.map(point => point[1]).reduce((total, current) => total += current)/currentCluster.length;
    shiftedCentroids.push([centroidUpdateX, centroidUpdateY]);
    currentCluster.forEach(function(point){
      clusteredCentroids.push([point, [centroidUpdateX, centroidUpdateY], i]);
    });
  };
  return [shiftedCentroids, clusteredCentroids];
};

// Find the indices of the nearest centroids for each of points
var nearestIndex = function(points, centroids) {
  // Initialize array of centroid indices, i.e. cluster labels
  var indexArray = [];
  for(var i=0; i < points.length; i++){
    indexArray.push(nearestCentroid(points[i], centroids));
  };
  return indexArray;
};

var dataset;
var maxSteps = 3;
var threshold = 0.1;

// Load data
var a1 = d3.csv("../../../../../datasets/elements/Unsupervised/Cluster_Analysis/kmeans/a0.csv", rowConverter);



// Create scatter plot
a1.then(function(data) {

// assign data to variable for access
dataset = data;

// Create scale functions
var xScale = d3.scaleLinear()
               .domain([d3.min(data, function(d) { return d[0]; }) - padding, d3.max(data, function(d) { return d[0]; }) + padding])
               .range([0, w]);

var yScale = d3.scaleLinear()
               .domain([d3.min(data, function(d) { return d[1]; })- padding, d3.max(data, function(d) { return d[1]; }) + padding])
               .range([h, 0]);
// Create line drawing function
var line = d3.line()
             .x(function(d){ return xScale(d[0]); })
             .y(function(d){ return yScale(d[1]); });

 // Define X axis
var xAxis = d3.axisBottom()
              .scale(xScale)
              .ticks(5);

// Define Y axis
var yAxis = d3.axisLeft()
              .scale(yScale)
              .ticks(5);

// Create SVG element
var svg = d3.select("#top-plot")
            .append("svg")
              .attr("width", w + margin.right + margin.left)
              .attr("height", h + margin.top + margin.bottom)
            .append("g")
              .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Create x-axis
svg.append("g")
   .attr("class", "axis")
   .attr("transform", "translate(0," + (h ) + ")")
   .call(xAxis);

// Create y-axis
svg.append("g")
  .attr("class", "axis")
  .attr("transform", "translate(" + 0 + ",0)")
  .call(yAxis);

// Step 1

// Draw points of scatter plot
var circles = svg.append("g")
                   .attr("class", "points")
                 .selectAll("circle")
                 .data(data)
                 .enter()
                 .append("circle")
                   .attr("class", "point")
                   .attr("cx", function(d) {
                    return xScale(d[0]);
                   })
                   .attr("cy", function(d) {
                    return yScale(d[1]);
                   })
                   .attr("r", 1)
                   .attr("fill", "black");
// Step 2

// Set initial cluster centroids
var [initIndex, currentCentroids] = initalizeCentroids(dataset, k);

// Step 3

// Add and Highlight initial centroids
var centroids = svg.append("g")
                     .attr("class", "centroids")
                   .selectAll("circle")
                   .data(currentCentroids)
                   .enter()
                   .append("circle")
                     .attr("class", "centroid")
                     .attr("cx", function(d) {
                       return xScale(d[0]);
                     })
                     .attr("cy", function(d) {
                       return yScale(d[1]);
                     })
                     .transition("centroid-init")
                     .duration(drawInitCentroids)
                     .attr("r", 4)
                     .attr("fill", function(d, i){
                       return colors(i);
                     })
                     .on("end", function() {
                       d3.select(this)
                       .transition("centroid-fade")
                       .delay(drawClusterAssignmentLines)
                       .duration(colorClusterAssignment)
                       .attr("fill", "grey")
                       .attr("r", 2);
                     });


// Step 4

// Find the index of the nearest centroid to each point
var currentIndices = nearestIndex(data, currentCentroids);
// Create array combining data points and centroids
var assignedPoints = assignCluster(data, currentCentroids, currentIndices);


// Step 5

// Draw a line connecting each point to the nearest centroid
var lines = svg.append("g")
                 .attr("class", "lines")
               .selectAll("line")
               .data(assignedPoints)
               .enter()
               .append("line")
                 .attr("class", "length")
                 .attr("x1", function(d) { return xScale(d[0][0]); })
                 .attr("y1", function(d) { return yScale(d[0][1]); })
                 .attr("x2", function(d) { return xScale(d[1][0]); })
                 .attr("y2", function(d) { return yScale(d[1][1]); });

            lines.transition("stroke-add")
                 .delay(drawInitCentroids)
                 .duration(drawClusterAssignmentLines)
                 .attr("stroke-width", 1)
                 .attr("stroke", function(d){
                   return colors(d[2]);
                 })
                 .attr("stroke-opacity", 0.1)
                 .on("end", function() {
                   d3.select(this)
                     .transition("stroke-update")
                     .duration(grayClusterAssignmentLines)
                       .attr("stroke", "grey")
                       .attr("stroke-opacity", 0.05)
                     .on("end", function() {
                       d3.select(this)
                         .transition("stroke-fade")
                         .duration(fadeClusterAssignmentLines)
                           .attr("stroke-opacity", 0);
                       });
                 });

 // Step 6

 // Change color of points to match assigned cluster centroid
 circles.data(assignedPoints)
        .transition("cluster-update")
        .delay(waitClusterAssignment)
        .duration(colorClusterAssignment)
        .attr("fill", function(d, i){
          return colors(d[2]);
        });



        // Step 7
        // Compute new centroids
        var [newCentroids, generatingClusters] = centroidUpdate(assignedPoints);

        // Draw a line connecting each point to the new centroid
        svg.selectAll(".length")
           .data(generatingClusters)
           .transition("stroke-move")
           .delay(waitCentroidAssignmentLines)
           .duration(moveCentroidAssignmentLines)
           .attr("x1", function(d) { return xScale(d[0][0]); })
           .attr("y1", function(d) { return yScale(d[0][1]); })
           .attr("x2", function(d) { return xScale(d[1][0]); })
           .attr("y2", function(d) { return yScale(d[1][1]); })
           .on("end", function(){
             d3.select(this)
               .transition("stroke-centroid")
               .duration(drawCentroidAssignmentLines)
               .attr("stroke", function(d){
                 return colors(d[2]);
               })
               .attr("stroke-opacity", 0.1)
               .on("end", function(){
                 d3.select(this)
                   .transition("stroke-inject")
                   .delay(moveCentroidShift)
                   .duration(fillCentroidShift)
                   .attr("stroke", "grey")
                   .attr("stroke-opacity", 0.05)
                   .on("end", function(){
                     d3.select(this)
                       .transition("post-inject-fade")
                       .duration(500)
                       .attr("stroke-opacity", 0);
                   });
               });
           });

        // Step 8
        // Move centroids to new positions
        svg.selectAll(".centroid")
           .data(newCentroids)
           .transition("centroid-shift")
           .delay(waitCentroidShift)
           .duration(moveCentroidShift)
           .attr("cx", function(d) {
             return xScale(d[0]);
           })
           .attr("cy", function(d) {
             return yScale(d[1]);
           })
           .on("end", function(d, i) {
             d3.select(this)
               .transition("centroid-fill")
               .duration(fillCentroidShift)
               .attr("fill", function(){
                 return colors(i);
               })
               .attr("r", 4);
           });


        // Drain color of points after computing new centroid
        circles.data(assignedPoints)
               .transition("cluster-update")
               .delay(waitClusterDecompose)
               .duration(fillCentroidShift)
               .attr("fill", "black");

        currentCentroids = newCentroids;
/*
        currentIndices = nearestIndex(data, newCentroids);
        assignedPoints = assignCluster(data, newCentroids, currentIndices);

        // Draw a line connecting each point to the nearest centroid
        svg.selectAll(".length")
           .data(assignedPoints)
           .attr("x1", function(d) { return xScale(d[0][0]); })
           .attr("y1", function(d) { return yScale(d[0][1]); })
           .attr("x2", function(d) { return xScale(d[1][0]); })
           .attr("y2", function(d) { return yScale(d[1][1]); })
           .transition("stroke-add")
           .delay(drawClusterAssignmentLines + shiftCentroidPosition)
           .duration(drawClusterAssignmentLines)
           .attr("stroke-width", 1)
           .attr("stroke", function(d){
             return colors(d[2]);
           })
           .attr("stroke-opacity", 0.1)
           .on("end", function() {
             d3.select(this)
               .transition("stroke-update")
               .duration(grayClusterAssignmentLines)
                 .attr("stroke", "grey")
                 .on("end", function() {
                   d3.select(this)
                     .transition("stroke-fade")
                     .duration(fadeClusterAssignmentLines)
                       .attr("stroke-opacity", 0);
                   });
           });

           // Change color of points to match assigned cluster centroid
           circles.data(assignedPoints)
                  .transition("cluster-update")
                  .delay(drawClusterAssignmentLines + waitClusterAssignment)
                  .duration(colorClusterAssignment)
                  .attr("fill", function(d, i){
                    return colors(d[2]);
                  });
*/


// Update loop
var step = 0;

var t = d3.interval(function(){

          console.time("loop");

          svg.selectAll(".centroid")
             .data(currentCentroids)
               .transition("centroid-fade")
               .delay(waitClusterAssignment)
               .duration(colorClusterAssignment)
             .attr("cx", function(d) {
               return xScale(d[0]);
             })
             .attr("cy", function(d) {
               return yScale(d[1]);
             })
             .attr("fill", "grey")
             .attr("r", 2);


          // Step 4

          // Find the index of the nearest centroid to each point
          var currentIndices = nearestIndex(data, currentCentroids);
          // Create array combining data points and centroids
          var assignedPoints = assignCluster(data, currentCentroids, currentIndices);


          // Step 5

          // Draw a line connecting each point to the nearest centroid

                      lines.data(assignedPoints)
                           .attr("x1", function(d) { return xScale(d[0][0]); })
                           .attr("y1", function(d) { return yScale(d[0][1]); })
                           .attr("x2", function(d) { return xScale(d[1][0]); })
                           .attr("y2", function(d) { return yScale(d[1][1]); })
                           .transition("stroke-add")
                           .delay(drawInitCentroids)
                           .duration(drawClusterAssignmentLines)
                           .attr("stroke-width", 1)
                           .attr("stroke", function(d){
                             return colors(d[2]);
                           })
                           .attr("stroke-opacity", 0.1)
                           .on("end", function() {
                             d3.select(this)
                               .transition("stroke-update")
                               .duration(grayClusterAssignmentLines)
                                 .attr("stroke", "grey")
                                 .attr("stroke-opacity", 0.05)
                               .on("end", function() {
                                 d3.select(this)
                                   .transition("stroke-fade")
                                   .duration(fadeClusterAssignmentLines)
                                     .attr("stroke-opacity", 0);
                                 });
                           });

           // Step 6

           // Change color of points to match assigned cluster centroid
           circles.data(assignedPoints)
                  .transition("cluster-update")
                  .delay(waitClusterAssignment)
                  .duration(colorClusterAssignment)
                  .attr("fill", function(d, i){
                    return colors(d[2]);
                  });



                  // Step 7
                  // Compute new centroids
                  var [newCentroids, generatingClusters] = centroidUpdate(assignedPoints);

                  // Draw a line connecting each point to the new centroid
                  svg.selectAll(".length")
                     .data(generatingClusters)
                     .transition("stroke-move")
                     .delay(waitCentroidAssignmentLines)
                     .duration(moveCentroidAssignmentLines)
                     .attr("x1", function(d) { return xScale(d[0][0]); })
                     .attr("y1", function(d) { return yScale(d[0][1]); })
                     .attr("x2", function(d) { return xScale(d[1][0]); })
                     .attr("y2", function(d) { return yScale(d[1][1]); })
                     .on("end", function(){
                       d3.select(this)
                         .transition("stroke-centroid")
                         .duration(drawCentroidAssignmentLines)
                         .attr("stroke", function(d){
                           return colors(d[2]);
                         })
                         .attr("stroke-opacity", 0.1)
                         .on("end", function(){
                           d3.select(this)
                             .transition("stroke-inject")
                             .delay(moveCentroidShift)
                             .duration(fillCentroidShift)
                             .attr("stroke", "grey")
                             .attr("stroke-opacity", 0.05)
                             .on("end", function(){
                               d3.select(this)
                                 .transition("post-inject-fade")
                                 .duration(500)
                                 .attr("stroke-opacity", 0);
                             });
                         });
                     });

                  // Step 8
                  // Move centroids to new positions
                  svg.selectAll(".centroid")
                     .data(newCentroids)
                     .transition("centroid-shift")
                     .delay(waitCentroidShift)
                     .duration(moveCentroidShift)
                     .attr("cx", function(d) {
                       return xScale(d[0]);
                     })
                     .attr("cy", function(d) {
                       return yScale(d[1]);
                     })
                     .on("end", function(d, i) {
                       d3.select(this)
                         .transition("centroid-fill")
                         .duration(fillCentroidShift)
                         .attr("fill", function(){
                           return colors(i);
                         })
                         .attr("r", 4);
                     });


                  // Drain color of points after computing new centroid
                  circles.data(assignedPoints)
                         .transition("cluster-update")
                         .delay(waitClusterDecompose)
                         .duration(fillCentroidShift)
                         .attr("fill", "black");





  console.log("currentCentroids");
  console.log(currentCentroids);


     step += 1

     var centroidShifts = [];
     for(var i = 0; i < k; i++) {
       centroidShifts.push(squaredError(newCentroids[i], currentCentroids[i]));
     };
     var maxShift = d3.max(centroidShifts);

     console.log("centroidShifts");
     console.log(centroidShifts);

     console.log("maxShift");
     console.log(maxShift);
     (maxShift < threshold) ? t.stop() : currentCentroids = newCentroids;

     console.log("newCentroids");
     console.log(newCentroids);

    //if (step > maxSteps) t.stop();

    console.timeEnd("loop");
  //
}, drawClusterAssignmentLines + updateTime, 0);

});
