'use strict';

const expect = chai.expect;

describe('test MLContext', async function() {
  it('navigator.ml should be a ML', function CheckML() {
    if (typeof window !== 'undefined') {
      expect(navigator.ml).to.be.an.instanceof(ML);
    } else {
      // This case does not need to be tested on Node.js
      this.skip();
    }
  });

  it('ml.createContext should be a function', () => {
    expect(navigator.ml.createContext).to.be.a('function');
  });

  it('ml.createContext should return a MLContext', () => {
    expect(navigator.ml.createContext({type: 'webnn', devicePreference: 'gpu'})).to.be.an.instanceof(MLContext);
  });

  it('ml.createContext should support MLContextOptions', async () => {
    expect(navigator.ml.createContext({})).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContext({
      powerPreference: 'default',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContext({
      powerPreference: 'low-power',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContext({
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(MLContext);
  });

  it('ml.createContext should throw for invalid options', async () => {
    expect(() => navigator.ml.createContext('invalid')).to.throw(Error);
  });

  it('ml.createContext should throw for invalid power preference', async () => {
    expect(() => navigator.ml.createContext({
      powerPreference: 'invalid',
    })).to.throw(Error);
  });
});
