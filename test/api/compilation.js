'use strict';

const expect = chai.expect;
const assert = chai.assert;

import * as utils from '../utils.js';

describe('test Compilation', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();
  const desc = {type: 'float32', dimensions: [2, 2]};
  const a = builder.input('a', desc);
  const b = builder.input('b', desc);
  const c = builder.matmul(a, b);
  const d = builder.constant(
      {type: 'float32', dimensions: [2, 2]}, new Float32Array(4).fill(1));
  const e = builder.add(c, d);
  const bufferA = new Float32Array(4).fill(1);
  const bufferB = new Float32Array(4).fill(1);
  const expectedC = [2, 2, 2, 2];
  const expectedE = [3, 3, 3, 3];

  it('Compilation should have compute method', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    expect(compiledModel.compute).to.be.a('function');
  });

  it('Compilation.compute should accept inputs and return outputs',
     async () => {
       const model = builder.createModel({c});
       const compiledModel = await model.compile();
       const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
       const outputs = await compiledModel.compute(inputs);
       expect(outputs).to.be.a('object');
       expect(outputs).to.have.property('c');
       expect(outputs.c).to.have.property('buffer');
       utils.checkValue(outputs.c.buffer, expectedC);
       expect(outputs.c).to.have.property('dimensions');
       utils.checkShape(outputs.c.dimensions, [2, 2]);
     });

  it('Compilation.compute should accept inputs and pre-allocated outputs',
     async () => {
       const model = builder.createModel({c});
       const compiledModel = await model.compile();
       const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
       const bufferC = new Float32Array(4);
       const outputs = {c: {buffer: bufferC}};
       await compiledModel.compute(inputs, outputs);
       utils.checkValue(bufferC, expectedC);
     });

  it('Compilation.compute should support multiple outputs', async () => {
    const model = builder.createModel({c, e});
    const compiledModel = await model.compile();
    const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
    const outputs = await compiledModel.compute(inputs);
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('c');
    expect(outputs.c).to.have.property('buffer');
    utils.checkValue(outputs.c.buffer, expectedC);
    expect(outputs.c).to.have.property('dimensions');
    utils.checkShape(outputs.c.dimensions, [2, 2]);
    expect(outputs).to.have.property('e');
    expect(outputs.e).to.have.property('buffer');
    utils.checkValue(outputs.e.buffer, expectedE);
    expect(outputs.e).to.have.property('dimensions');
    utils.checkShape(outputs.e.dimensions, [2, 2]);
  });

  it('Compilation.compute should support specified outputs', async () => {
    const model = builder.createModel({c, e});
    const compiledModel = await model.compile();
    const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
    let outputs = await compiledModel.compute(inputs, {c});
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('c');
    expect(outputs.c).to.have.property('buffer');
    utils.checkValue(outputs.c.buffer, expectedC);
    expect(outputs.c).to.have.property('dimensions');
    utils.checkShape(outputs.c.dimensions, [2, 2]);
    expect(outputs).not.to.have.property('e');
    outputs = await compiledModel.compute(inputs, {e});
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('e');
    expect(outputs.e).to.have.property('buffer');
    utils.checkValue(outputs.e.buffer, expectedE);
    expect(outputs.e).to.have.property('dimensions');
    utils.checkShape(outputs.e.dimensions, [2, 2]);
    expect(outputs).not.to.have.property('c');
  });

  const descX = {type: 'float32', dimensions: [-1, 2]};
  const descY = {type: 'float32', dimensions: [2, -1]};
  const x = builder.input('x', descX);
  const y = builder.input('y', descY);
  const z = builder.matmul(x, y);
  it('Compilation.compute should support inputs with specified shape',
     async () => {
       const model = builder.createModel({z});
       const compiledModel = await model.compile();
       const shapeX = [3, 2];
       const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
       const shapeY = [2, 4];
       const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
       const inputs = {
         x: {buffer: bufferX, dimensions: shapeX},
         y: {buffer: bufferY, dimensions: shapeY},
       };
       const outputs = await compiledModel.compute(inputs);
       expect(outputs).to.be.a('object');
       expect(outputs).to.have.property('z');
       expect(outputs.z).to.have.property('buffer');
       const shapeZ = [shapeX[0], shapeY[1]];
       const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
       utils.checkValue(outputs.z.buffer, expectedZ);
       expect(outputs.z).to.have.property('dimensions');
       utils.checkShape(outputs.z.dimensions, shapeZ);
     });

  it('Compilation.compute should throw for non inputs', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for empty inputs', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute({});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for invalid input name', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute({x: {buffer: bufferA}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for missing input', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute({a: {buffer: bufferA}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for no input buffer', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute({a: {}, b: {}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for invalid input buffer', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      await compiledModel.compute({a: {buffer: 1}, b: {buffer: 2}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for invalid input dimensions',
     async () => {
       const model = builder.createModel({c});
       const compiledModel = await model.compile();
       try {
         await compiledModel.compute(
             {a: {buffer: bufferA, dimensions: [2]}, b: {buffer: bufferB}});
         assert.fail();
       } catch (err) {
         assert(!(err instanceof chai.AssertionError), 'No throwing');
         expect(err).to.be.an.instanceof(Error);
       }
     });

  it('Compilation.compute should throw for no dimensions for dynamic shape',
     async () => {
       const model = builder.createModel({z});
       const compiledModel = await model.compile();
       const shapeX = [3, 2];
       const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
       const shapeY = [2, 4];
       const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
       const inputs = {x: {buffer: bufferX}, y: {buffer: bufferY}};
       try {
         await compiledModel.compute(inputs);
         assert.fail();
       } catch (err) {
         assert(!(err instanceof chai.AssertionError), 'No throwing');
         expect(err).to.be.an.instanceof(Error);
       }
     });

  it('Compilation.compute should throw for invalid output name', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
      const bufferC = new Float32Array(4);
      const outputs = {z: {buffer: bufferC}};
      await compiledModel.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for invalid output buffer', async () => {
    const model = builder.createModel({c});
    const compiledModel = await model.compile();
    try {
      const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
      const outputs = {c: {buffer: []}};
      await compiledModel.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Compilation.compute should throw for invalid output buffer length',
     async () => {
       const model = builder.createModel({c});
       const compiledModel = await model.compile();
       try {
         const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
         const bufferC = new Float32Array(1);
         const outputs = {c: {buffer: bufferC}};
         await compiledModel.compute(inputs, outputs);
         assert.fail();
       } catch (err) {
         assert(!(err instanceof chai.AssertionError), 'No throwing');
         expect(err).to.be.an.instanceof(Error);
       }
     });
});
