const app = require('express')();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const fs = require('fs');

const NeuralNetwork = require('./neuralnetwork.js');
const Neat = require('./neat.js');

var speed = 0.5;
    
var dots = [];

var GAMES = 256;
var GAMES_PER_GENERATION = 20;
var GENERATIONS_PER_TRAINING = 20;
var SECONDS_PER_GAME = 5;
var PLAYER_COUNT = 2;
var TAGGER_COUNT = 1;

var ticks = 0;
var generations = 0;
var games = [];
var dt = 1/20;
var training = 0;

var writtenPlayers = [];
var players = [];
var baseNetwork = NeuralNetwork.createHiddenLayeredNetwork(PLAYER_COUNT*2, 4, 4);
for (var i = 0; i < PLAYER_COUNT; i++) {
    players.push(new Neat(GAMES, 4, baseNetwork));
    players[i].evolveRate = 0.9;
}

app.get('/', function(req, res){
    res.send('Server up and running :D');
});

io.on('connection', function(socket) {
    socket.emit('players', writtenPlayers);
    socket.emit('msg', (training?'Red':'Blue') + ' dot has been trained');
});

function newGeneration() {
    if (generations % GAMES_PER_GENERATION == 0) {
        //console.log("New Generation", Math.floor(generations/GAMES_PER_GENERATION) ,"of", training);
        players[training].evolve();
    }
    if (generations % (GENERATIONS_PER_TRAINING*GAMES_PER_GENERATION) == 0) {
        training = (training + 1) % PLAYER_COUNT;
        console.log("Begin training of", training);
        io.emit('msg', (training?'Red':'Blue') + ' dot has been trained');
        writeBestToDisk();
    }
    ticks = 0;
    games = [];
    for (var j = 0; j < GAMES; j++) {
        dots = [];
        for (var i = 0; i < players.length; i++) {
            var x = Math.random();
            var y = Math.random();
            dots.push({
                x: x, 
                y: y, 
                tagged: false, 
                tagger: i < TAGGER_COUNT
            });
        }
        games.push({dots: dots, moves: 0, toMove: true});
    }
    run();
}

function finishGeneration() {
    for (var j = 0; j < GAMES; j++) {
        dots = games[j].dots;
        var dot = dots[training];
        if (!dot.tagger && !dot.tagged)
            players[training].scores[j] += 1;
    }
    
    generations++;

    if (generations % GAMES_PER_GENERATION == 0) {
        //console.log("Finish Generation of", training);
        players[training].breed();
    }
    
    setTimeout(newGeneration, 1);
}

function runDot(game, i) {
    var dot = dots[i];
    if (!dot.tagged) {
        var network = players[i].networks[training == i ? game : 0];

        for (var j = 0; j < dots.length; j++) {
            network.setInputValue(j*2, dots[j].x);
            network.setInputValue(j*2 + 1, dots[j].y);
            //network.setInputValue(j*3 + 2, dots[j].tagged ? 1 : 0);
        }

        network.run();

        var up = network.getOutputValue(0);
        var down = network.getOutputValue(1);
        var left = network.getOutputValue(2);
        var right = network.getOutputValue(3);
        
        var lx = dot.x;
        var ly = dot.y;
        
        if (up) dot.y -= speed*dt;
        if (down) dot.y += speed*dt;
        if (left) dot.x -= speed*dt;
        if (right) dot.x += speed*dt;

        if (dot.x < 0) dot.x = 0;
        if (dot.x > 1) dot.x = 1;
        if (dot.y < 0) dot.y = 0;
        if (dot.y > 1) dot.y = 1;
        
        if (dot.x != lx || dot.y != ly) games[game].moves += 1;
    
        if (dot.tagger) {
            for (var j = 0; j < dots.length; j++) {
                var d2 = dots[j];
                if (!d2.tagger && !d2.tagged && 
                        Math.abs(dot.x - d2.x) < 0.05 && 
                        Math.abs(dot.y - d2.y) < 0.05) {
                    d2.tagged = true;
                    if (i == training)
                        players[i].scores[game] += 1;
                }
            }
        }
    }
}

function run() {
    dt = 1/20;
    ticks += dt;

    if (ticks >= SECONDS_PER_GAME) {
        finishGeneration();
        return;
    }

    for (var j = 0; j < GAMES; j++) {
        if (games[j].toMove) {
            games[j].moves = 0;
            dots = games[j].dots;
            for (var i = 0; i < dots.length; i++) {
                runDot(j, i);
            }
            games[j].toMove = games[j].moves > 0;
        }
    }
    run();
}

function writeBestToDisk() {
    writtenPlayers = [];
    for (var i = 0; i < players.length; i++) {
        players[i].networks[0].reset();
        writtenPlayers.push(players[i].networks[0].neurons);
    }
    fs.writeFile("players.json", JSON.stringify(writtenPlayers), (err) => {
        if (err) console.log(err);
    });
    
    io.emit('players', writtenPlayers);
}

fs.readFile("players.json", "utf-8", (err, data) => {
    if (err) { 
        console.log(err);
    } else {
        var loadPlayers = JSON.parse(data);
        players = [];
        for (var i = 0; i < loadPlayers.length; i++) {
            var network = NeuralNetwork.createHiddenLayeredNetwork(PLAYER_COUNT*2, 4, 4);
            network.neurons = loadPlayers[i];
            players.push(new Neat(GAMES, 4, network));
        }
    }
    
    newGeneration();
})

http.listen(1612, function(){
    console.log('Listening on *:1612');
});